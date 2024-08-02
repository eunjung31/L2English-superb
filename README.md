This repository provides data preparation codes for L2 English pronunciation scoring tasks in [[Dynamic Superb]](https://github.com/dynamic-superb/dynamic-superb), which assess accuracy aspect, fluency aspect, and prosodic aspect.

The repository is based on the public repository: [[linguistic-superb]](https://github.com/juice500ml/linguistic-superb). It includes data preparation codes for a set of tasks that are based on various schools of linguistics. The tasks include, but are not limited to, nonce word detection, phone feature classification, phone/phoneme counting, pronunciation similarity, Part of Speech (PoS) tagging, prosody naturalness, semantic textual similarity, and sentence grammar acceptability.


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

