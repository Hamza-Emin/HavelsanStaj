import csv
import json


csv_file_path = 'output.csv'
json_file_path = 'output.json'


fieldnames = ['Index', 'Explanation', 'Target', 'Clean', 'Aspects']

data = []
with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
    csv_reader = csv.DictReader(csv_file, fieldnames=fieldnames, delimiter=';')
    next(csv_reader)  
    for row in csv_reader:
       
        row['Aspects'] = json.loads(row['Aspects'].replace("'", "\""))
        data.append({
            'Explanation': row['Explanation'],
            'Target': row['Target'],
            'Clean': row['Clean'],
            'Aspects': row['Aspects']
        })


with open(json_file_path, mode='w', encoding='utf-8') as json_file:
    json.dump(data, json_file, indent=4, ensure_ascii=False)

print(f"CSV data has been converted to JSON and saved as {json_file_path}")
