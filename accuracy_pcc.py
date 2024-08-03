import os
import numpy as np
import random
from datasets import load_dataset, Dataset, Audio
from collections import defaultdict
from typing import Dict
from utils import validate_dataset
import itertools

first_parts = [
    "Listen to the audio and assess the clarity and accuracy of pronunciation by considering precise phonology, pronunciation errors, and overall comprehensibility.",
    "Evaluate the audio for clarity and pronunciation correctness, focusing on accurate phonology, any pronunciation mistakes, and general understandability.",
    "Listen to the audio recording and judge the clarity and correctness of pronunciation, taking into account accurate phonology, the presence of errors, and overall understandability.",
    "Assess the audio for pronunciation clarity and correctness, noting accurate phonology, pronunciation errors, and overall comprehensibility.",
    "Listen to the audio and rate the clarity and accuracy of pronunciation, considering accurate phonology, any errors in pronunciation, and overall understandability.",
    "Evaluate the clarity and correctness of the audio's pronunciation by considering precise phonology, pronunciation errors, and general understandability.",
    "Listen to the audio and determine the clarity and correctness of pronunciation by focusing on accurate phonology, errors, and overall comprehensibility.",
    "Assess the audio for pronunciation clarity and accuracy, considering accurate phonology, the presence of errors, and overall understandability.",
    "Listen to the audio recording and evaluate the clarity and correctness of pronunciation, taking into account accurate phonology, pronunciation mistakes, and general comprehensibility.",
    "Evaluate the audio for clarity and pronunciation correctness by considering precise phonology, any errors, and overall understandability."
]

second_parts = [
    "Provide your answer using Arabic numerals from 0 to 10.",
    "Respond with an Arabic numeral between 0 and 10.",
    "Give your answer in Arabic numerals from 0 to 10.",
    "Use an Arabic numeral in the range of 0 to 10 for your response.",
    "Answer with an Arabic numeral from 0 to 10.",
    "Respond using an Arabic numeral between 0 and 10.",
    "Provide a numerical answer in Arabic numerals from 0 to 10.",
    "Use an Arabic numeral from 0 to 10 for your answer.",
    "Give your response as an Arabic numeral from 0 to 10.",
    "Answer with an Arabic numeral within the range of 0 to 10."
]

# Combine them using a list comprehension
instructions = [f + " " + s for f, s in itertools.product(first_parts, second_parts)]

## Convert rows to a dataset
def rows_to_dataset(rows: Dict[str, list]) -> Dataset:
    ds = Dataset.from_dict(rows)
    for key in rows.keys():
        if "audio" in key:
            ds = ds.cast_column(key, Audio(sampling_rate=16000))
    return ds

if __name__ == "__main__":
    ds = load_dataset("mispeech/speechocean762",
            cache_dir="speechocean762",
                    #   cache_dir="/data/user_data/eyeo2",
                      num_proc=64)
    new_ds = ds["test"]

    ## exclude score 0 which indicates "no voice".
    new_ds = new_ds.filter(lambda x: x["accuracy"] != 0)

    # Categorize the samples by labels
    length_categories = defaultdict(list)
    for i, sample in enumerate(new_ds):
        length_categories[sample["accuracy"]].append(i)

    # Randomly select 30% of the samples for each label
    selected_indices = []
    for i, (length, indices) in enumerate(sorted(length_categories.items())):
        num_to_select = int(len(indices) * 0.3)
        random.seed(i)
        selected_indices.extend(random.sample(indices, num_to_select))

    # Create a new dataset with the selected indices
    new_ds = new_ds.select(selected_indices)
    print(len(new_ds))

    # Reformatting
    def _map(sample, index):
        return {
            "audio": sample["audio"],
            "file": sample["audio"]["path"].replace('.wav', ''),
            "instruction": instructions[index % len(instructions)],
            "label": str(sample["accuracy"]),
        }
    new_ds = new_ds.map(_map, with_indices=True, remove_columns=ds["test"].column_names)
    new_ds = new_ds.cast_column("audio", Audio(sampling_rate=16_000))

    # Push to Hugging Face
    validate_dataset(new_ds)
    new_ds.push_to_hub(repo_id="DynamicSuperb/L2EnglishAccuracy_speechocean762-Scoring", split="test", token=os.environ["HF_TOKEN"])
