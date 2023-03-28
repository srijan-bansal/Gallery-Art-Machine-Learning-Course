from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import concatenate_datasets
import evaluate
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
nltk.download("punkt")
import torch 

torch.manual_seed(3)


# helper function to postprocess text
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels


def main():

    dataset = load_dataset("csv", data_files="data.csv")
    dataset = dataset['train']  
    context = " [SEP] "
    poem_id = "-1"
    updated_input = []
    # add context

    for ind in range(len(dataset['id'])):
        curr_id = dataset['id'][ind]
        if poem_id != curr_id:
            context = " [SEP] "
            poem_id = curr_id
        else:
            context += " " + dataset['stanza'][ind].replace('\n', ' ')

        updated_input.append(dataset['explanation'][ind] + context)
    
    dataset = dataset.add_column("context", updated_input)

    dataset = dataset.train_test_split(0.1)
    print(f"Train dataset size: {len(dataset['train'])}")
    print(f"Test dataset size: {len(dataset['test'])}")

    model_id= "t5-large" #"google/flan-t5-base"
    
    # Load tokenizer of FLAN-t5-base
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    metric = evaluate.load("rouge")    

    # The maximum total input sequence length after tokenization. 
    # Sequences longer than this will be truncated, sequences shorter will be padded.
    tokenized_inputs = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x["context"], truncation=True), batched=True, remove_columns=["id", "title", "url", "stanza", "explanation", "context"])    
    max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]]) + 10
    print(f"Max source length: {max_source_length}")

    # The maximum total sequence length for target text after tokenization. 
    # Sequences longer than this will be truncated, sequences shorter will be padded."
    tokenized_targets = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x["stanza"], truncation=True), batched=True, remove_columns=["id", "title", "url", "stanza", "explanation", "context"])
    max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])
    print(f"Max target length: {max_target_length}")
    
    def preprocess_function(sample, padding="max_length", max_source_length=256, max_target_length=512):
        # add prefix to the input for t5
        prompt = "Write a poem: "
        
        inputs = [f"{prompt}: " + item for item in sample["context"]]

        # tokenize inputs
        model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        sample["stanza"] = [item.replace('\n',' ') for item in sample["stanza"]]
        labels = tokenizer(text_target=sample["stanza"], max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs    
    
    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["id", "title", "url", "stanza", "explanation", "context"])
    print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")
    
    # we want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8
    )
    

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    # Define training args
    training_args = Seq2SeqTrainingArguments(
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        fp16=False, # Overflows with fp16
        learning_rate=5e-5,
        num_train_epochs=10,
        logging_dir="logs",
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_rouge1",
        report_to="tensorboard",
        output_dir="output",
        push_to_hub=False
    )

    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=compute_metrics,
    )    

    trainer.train()
    trainer.evaluate()    
    
    def predict(user_explanation):
        prompt = "Write a poem: "
        inputs = [f"{prompt}: " + user_explanation]
        model_inputs = tokenizer(inputs, max_length=max_source_length, truncation=True, return_tensors="pt").input_ids.to('cuda')

        print(tokenizer.decode(model.generate(model_inputs)[0], skip_special_tokens=True))
        
        
    user_explanation = "The blue sky arches over the Yarrow Vale, with only a tender hazy brightness around the rising sun. It's a mild dawn of promise that eliminates all useless dejection. Though I am willing to allow a bit of pensive recollection here, this promise of a new day fills me with hope. [SEP] "
    predict(user_explanation)
    import pdb
    pdb.set_trace()    



    
main ()


