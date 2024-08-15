# For ranking task, different filename format causes errors. We need to change scores.json files to match the new filename format.

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

def modify_file(file_path, mapping):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    with open(file_path, 'w') as f:
        for line in lines:
            parts = line.strip().split('\t')
            old_id = parts[0].split('.')[0]
            if old_id in mapping:
                for new_id in mapping[old_id]:
                    parts[0] = new_id + '.' + parts[0].split('.')[1]
                    f.write('\t'.join(parts[:2]) + ' ' + ' '.join(parts[2:]) + '\n')

mapping = create_mapping('text')
modify_file('text-phone', mapping)
modify_file('text-phone.int', mapping)

def sort_file_by_first_column(file_path): 
    with open(file_path, 'r') as f: 
        lines = f.readlines()
        lines.sort(key=lambda line: line.split('\t')[0])

    with open(file_path, 'w') as f:
        f.writelines(lines)

sort_file_by_first_column('text-phone')
sort_file_by_first_column('text-phone.int')