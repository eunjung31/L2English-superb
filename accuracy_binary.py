import os
import numpy as np
import random
from datasets import load_dataset, Dataset, Audio
from collections import defaultdict
from typing import Dict
from utils import validate_dataset
import itertools

first_parts = [
    "Listen to both the first and second audio clips and compare their pronunciation accuracy.",
    "Compare the pronunciation accuracy of the first and second audio recordings.",
    "Listen to the first and second audio samples and evaluate their pronunciation accuracy.",
    "Assess the pronunciation accuracy of both the first and second audio clips.",
    "Listen to the first and second audio recordings and compare their pronunciation accuracy.",
]

second_parts = [
    "Evaluate the clarity and correctness of pronunciation by considering accurate phonology, presence of pronunciation errors, and overall understandability.",
    "Assess the clarity and correctness of pronunciation, taking into account accurate phonology, any pronunciation mistakes, and overall understandability.",
    "Focus on clarity, correct phonology, presence of pronunciation errors, and overall understandability.",
    "Evaluate the clarity, correctness of phonology, presence of any pronunciation mistakes, and overall understandability.",
    "Focus on evaluating the clarity, correct phonology, pronunciation mistakes, and overall understandability.",
]

third_parts = [
    "Answer the question: 'Is the pronunciation accuracy of the first audio higher than that of the second audio?' with 'Yes' if the first audio is better, and 'No' if the second audio is better.",
    "Answer the question: 'Does the first audio exhibit better pronunciation accuracy compared to the second audio?' with 'Yes' if the first audio is better, and 'No' if the second audio is better.",
    "Answer the question: 'Is the pronunciation accuracy of the first audio better than the second audio?' with 'Yes' if the first audio is better, and 'No' if the second audio is better.",
    "Answer the question: 'Does the first audio have better pronunciation accuracy than the second audio?' with 'Yes' if the first audio is better, and 'No' if the second audio is better.",
    "Answer the question: 'Is the pronunciation accuracy in the first audio better than in the second audio?' with 'Yes' if the first audio is better, and 'No' if the second audio is better."
]

# Combine them using a list comprehension
instructions = [f + " " + s + " " + t for f, s, t in itertools.product(first_parts, second_parts, third_parts)]

## utterance 0<=score<=3 is only 1. So we combine score 0~4 to score 1.
## score distribution: {1: 161, 2: 299, 3:1343, 4: 767}
def map_scores(score):
    if 0 <= score <= 4:
        return 1
    elif 5 <= score <= 6:
        return 2
    elif 7 <= score <= 8:
        return 3
    elif 9 <= score <= 10:
        return 4

## Reformatting and mapping scores.
def reformat_and_map(sample, index):
    sample["accuracy"] = map_scores(sample["accuracy"])
    return sample

## Filter samples with audio longer than 2.2 seconds
def filter_long_audio(sample):
    return sample["audio"]["array"].shape[0] / sample["audio"]["sampling_rate"] > 2.2

## Get balanced pairs of samples with different scores.
def get_balanced_pairs_and_counts(dataset, target_pairs_per_pair=30):
    print(len(dataset))
    # Convert the dataset to a pandas DataFrame if it is a Hugging Face Dataset
    if isinstance(dataset, Dataset):
        dataset = dataset.to_pandas()

    accuracy_values = [1, 2, 3, 4]
    pairs = [(a1, a2) for a1, a2 in itertools.product(accuracy_values, repeat=2) if a1 != a2]
    
    pair_counts = defaultdict(int)
    pairs_of_samples = []

    for a1, a2 in pairs:
        samples_a1 = dataset[dataset['accuracy'] == a1]
        samples_a2 = dataset[dataset['accuracy'] == a2]

        # Calculate the number of available samples for this pair
        count_a1 = len(samples_a1)
        count_a2 = len(samples_a2)
        pair_counts[(a1, a2)] = min(count_a1, count_a2)

    # Determine the minimum number of samples available across all pairs
    min_samples_per_pair = min(pair_counts.values())
    # Limit to the target pairs per pair if specified
    num_samples_per_pair = min(min_samples_per_pair, target_pairs_per_pair)

    selected_pair_counts = defaultdict(int)

    for a1, a2 in pairs:
        samples_a1 = dataset[dataset['accuracy'] == a1]
        samples_a2 = dataset[dataset['accuracy'] == a2]

        count = 0
        for _, sample1 in samples_a1.iterrows():
            for _, sample2 in samples_a2.iterrows():
                if count < num_samples_per_pair:
                    pairs_of_samples.append((sample1, sample2))
                    selected_pair_counts[(a1, a2)] += 1
                    count += 1
                else:
                    break
            if count >= num_samples_per_pair:
                break

    return pairs_of_samples, selected_pair_counts

## Convert rows to a dataset
def rows_to_dataset(rows: Dict[str, list]) -> Dataset:
    ds = Dataset.from_dict(rows)
    for key in rows.keys():
        if "audio" in key:
            ds = ds.cast_column(key, Audio(sampling_rate=16000))
    return ds

## Count utterances by score in a dataset
def count_utterances_by_score(dataset, score_keys=["score1", "score2"]):
    score_counts = defaultdict(int)
    for sample in dataset:
        for key in score_keys:
            score_counts[sample[key]] += 1
    return score_counts

if __name__ == "__main__":
    ds = load_dataset("mispeech/speechocean762",
                      cache_dir="/data/user_data/eyeo2",
                      num_proc=64)
    new_ds = ds["test"]

    # Reformatting and mapping scores
    new_ds = new_ds.map(reformat_and_map, with_indices=True)

    # Filter out samples with None accuracy
    new_ds = new_ds.filter(lambda x: x["accuracy"] is not None)

    # Filter out samples with audio shorter than 0.02 seconds
    new_ds = new_ds.filter(filter_long_audio)

    # Count utterances by score in the original dataset
    original_score_counts = count_utterances_by_score(new_ds, score_keys=["accuracy"])
    print("Original score counts:", dict(original_score_counts))

    # Get balanced pairs
    balanced_pairs, selected_pair_counts = get_balanced_pairs_and_counts(new_ds, target_pairs_per_pair=30)
    print(selected_pair_counts)

    # Create a new dataset with the selected pairs
    paired_data = []
    for sample1, sample2 in balanced_pairs:
        score1 = sample1["accuracy"]
        score2 = sample2["accuracy"]
        label = "Yes" if score1 > score2 else "No"
        
        paired_data.append({
            "audio1": sample1["audio"],
            "audio2": sample2["audio"],
            "file1": sample1["audio"]["path"],
            "file2": sample2["audio"]["path"],
            "instruction": instructions[random.randint(0, len(instructions) - 1)],
            "score1": score1,
            "score2": score2,
            "label": label
        })

    # Convert paired_data to a dataset
    paired_ds = rows_to_dataset({
        "audio": [item["audio1"] for item in paired_data],
        "audio2": [item["audio2"] for item in paired_data],
        "file": [item["file1"] + "_" + item["file2"] for item in paired_data],
        "instruction": [item["instruction"] for item in paired_data],
        "label": [item["label"] for item in paired_data]
    })

    # Shuffle the dataset
    paired_ds = paired_ds.shuffle(seed=42)

    print(len(paired_ds))

    # Push to Hugging Face
    validate_dataset(paired_ds)
    paired_ds.push_to_hub(repo_id="DynamicSuperb/L2EnglishAccuracy_speechocean762-BinaryAccuracy", split="test", token=os.environ["HF_TOKEN"])
