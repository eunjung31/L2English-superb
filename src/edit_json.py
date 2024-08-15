# For ranking task, different filename format causes errors. We need to change scores.json files to match the new filename format.
import json

def create_mapping(text_file):
    mapping = {}
    with open(text_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            old_id = parts[0].split('_')[1]
            new_id = parts[0]
            if old_id in mapping:
                mapping[old_id].append(new_id)
            else:
                mapping[old_id] = [new_id]
    return mapping

def modify_json(file_path, mapping):
    with open(file_path, 'r') as f:
        data = json.load(f)

    new_data = {}
    for old_id, value in data.items():
        if old_id in mapping:
            for new_id in mapping[old_id]:
                new_data[new_id] = value

    with open(file_path, 'w') as f:
        json.dump(new_data, f, indent=2, sort_keys=True)

mapping = create_mapping('text')
modify_json('scores.json', mapping)