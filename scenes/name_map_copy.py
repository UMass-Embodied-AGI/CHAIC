import os
import shutil
import json

all_data = json.load(open('dataset/name_map.json'))

def find_and_replace(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == 'name_map.json':
                file_path = os.path.join(root, file)
                print(f"Found: {file_path}")
                data = json.load(open(file_path))
                for key in data:
                    assert key in all_data and all_data[key] == data[key], f"Key: {key} not found or different value"
                    all_data[key] = data[key]

                # Replace
                with open(file_path, 'w') as f:
                    json.dump(all_data, f, indent=4)

directory_path = 'dataset'
find_and_replace(directory_path)

print(all_data)
print(len(all_data))