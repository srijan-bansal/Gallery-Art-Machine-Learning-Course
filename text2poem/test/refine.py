import json
import os

files = os.listdir('.')
for file in files:
    if 'json' in file:
        data = json.load(open(file,'r'))
        data_temp = []
        for item in data:
            data_temp.append({
                'id' : item['id'],
                'url': '',
                'text': item['text'],
                'poem_id': '',
                'poem_title': '',
                'explanation': item['explanation'],
            })
        json.dump(data_temp, open(f'refined/{file}','w'))
        
