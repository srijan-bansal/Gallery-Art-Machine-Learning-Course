---
tags:
- generated_from_trainer
model-index:
- name: results
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# results

This model is a fine-tuned version of [models/poemization-t5-large-br](https://huggingface.co/models/poemization-t5-large-br) on the None dataset.
It achieves the following results on the evaluation set:
- eval_loss: 2.2473
- eval_rouge1: 40.4676
- eval_rouge2: 13.3257
- eval_rougeL: 31.7504
- eval_rougeLsum: 32.3617
- eval_gen_len: 89.1667
- eval_runtime: 17.4245
- eval_samples_per_second: 0.689
- eval_steps_per_second: 0.172
- step: 0

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 4
- eval_batch_size: 4
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 16
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 10.0

### Framework versions

- Transformers 4.24.0
- Pytorch 1.13.0+cu117
- Datasets 2.6.1
- Tokenizers 0.13.2
