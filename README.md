The repository is based on the public repository: [[linguistic-superb]](https://github.com/juice500ml/linguistic-superb). It includes data preparation codes for a set of tasks that are based on various schools of linguistics. The tasks include, but are not limited to, nonce word detection, phone feature classification, phone/phoneme counting, pronunciation similarity, Part of Speech (PoS) tagging, prosody naturalness, semantic textual similarity, and sentence grammar acceptability.

## L2 English Pronunciation scoring
This repository provides data preparation codes for L2 English pronunciation scoring tasks in [[Dynamic Superb]](https://github.com/dynamic-superb/dynamic-superb). The task utilizes subdatasets of [[speechocean762 dataset]](https://github.com/jimbozhang/speechocean762), to meet the criteria of Dyanimc Superb (< 1 hour). The task includes three assessment tasks, which includes accuracy scoring, fluency scoring, and prosodic aspect scoring. Furthermore, we include two subtasks for each assessment task as follows: 
   - (1) pearson correlation coefficient (PCC) task: correlation between scores from provided human experts and predicted results
   - (2) binary accuracy task: ask the model to compare the scores of the two audios (e.g. which audio has higher scores?)


## Set up environment
```sh
conda create -p ./envs python=3.10
conda activate ./envs
pip install -r requirements.txt
```

## Push tasks
```sh
HF_TOKEN=YOUR_HF_TOKEN python3 TASK_NAME.py
```

## How to use utils
```python
from utils import rows_to_dataset, validate_dataset
import os

rows = {
    # audio file name has to be unique!
    "audio": ["/path/to/audio1.wav", "/path/to/audio2.wav", "/path/to/audio3.wav", ],
    "instruction": ["inst1", "inst2", "inst3"],
    "label": ["l1", "l2", "l3"],
}
ds = rows_to_dataset(rows)
validate_dataset(ds)
ds.push_to_hub(repo_id="your/repo_id", split="test", token=os.environ["HF_TOKEN"])
```

## References
[1] https://github.com/juice500ml/linguistic-superb
[2] 
