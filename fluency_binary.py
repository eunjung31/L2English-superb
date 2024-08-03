import os
import numpy as np
import random
from datasets import load_dataset, Dataset, Audio
from collections import defaultdict
from typing import Dict
from utils import validate_dataset
import itertools

first_parts = [
    "Listen to the audio and assess the smoothness and flow of speech, paying attention to pauses, repetitions, and stammering.",
    "Evaluate the audio for smoothness and speech flow, noting any pauses, repetitions, and stammering.",
    "Listen to the recording and focus on the smoothness and flow of speech, checking for instances of pauses, repetitions, and stammering.",
    "Assess the audio by concentrating on the smoothness and flow of speech, identifying any pauses, repetitions, and stammering.",
    "Evaluate the recording with an emphasis on the smoothness and flow of speech, observing any pauses, repetitions, and stammering.",
    "Listen to the audio, focusing on the smoothness and flow of the speech, and note any pauses, repetitions, and stammering.",
    "Assess the recording for smoothness and the flow of speech, looking out for pauses, repetitions, and stammering.",
    "Evaluate the speech in the audio, concentrating on its smoothness and flow, and checking for pauses, repetitions, and stammering.",
    "Listen and focus on the audio’s smoothness and speech flow, identifying any pauses, repetitions, and stammering.",
    "Assess the audio, emphasizing the smoothness and flow of speech, and note any pauses, repetitions, and stammering."
]

second_parts = [
    "Consider the smoothness and flow of speech, including any pauses, repetitions, and stammering.",
    "Focus on the smoothness and flow of speech, checking for pauses, repetitions, and stammering.",
    "Consider the smoothness and continuity of speech, including the presence of pauses, repetitions, and stammering.",
    "Take into account the smoothness and flow of speech, including any interruptions such as pauses, repetitions, and stammering.",
    "Focus on the flow and smoothness of speech, noting the occurrence of pauses, repetitions, and stammering.",
    "Consider how smoothly the speech flows, including the presence of pauses, repetitions, and stammering.",
    "Focus on the smoothness and continuity of speech, noting any pauses, repetitions, and stammering.",
    "Consider the flow and smoothness of speech, including pauses, repetitions, and stammering.",
    "Focus on the smoothness and flow of speech, checking for any interruptions such as pauses, repetitions, and stammering.",
    "Assess the clarity and correctness of pronunciation by considering accurate phonology, presence of pronunciation errors, and overall understandability."
]

third_parts = [
    "Answer the question: 'Does the first audio sample have better fluency than the second audio sample?' with 'Yes' if the first audio is better, and 'No' if the second audio is better.",
    "Respond to the question: 'Does the first audio sample exhibit better fluency compared to the second?' with 'Yes' if the first audio is better, and 'No' if the second audio is better.",
    "Answer the question: 'Is the fluency in the first audio sample higher than in the second?' with 'Yes' if the first audio is better, and 'No' if the second audio is better.",
    "Respond to the question: 'Is the fluency of the first audio sample better than the second?' with 'Yes' if the first audio is better, and 'No' if the second audio is better.",
    "Answer the question: 'Does the first audio sample have better fluency compared to the second?' with 'Yes' if the first audio is better, and 'No' if the second audio is better.",
]

# Combine them using a list comprehension
instructions = [f + " " + s + " " + t for f, s, t in itertools.product(first_parts, second_parts, third_parts)]

## utterance 0<=score<=3 is only 1. So we combine score 0~4 to score 1.
## score distribution: {1: 837, 2: 1258, 3:291, 4: 67}
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
    sample["fluency"] = map_scores(sample["fluency"])
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

    fluency_values = [1, 2, 3, 4]
    pairs = [(a1, a2) for a1, a2 in itertools.product(fluency_values, repeat=2) if a1 != a2]
    
    pair_counts = defaultdict(int)
    pairs_of_samples = []

    for a1, a2 in pairs:
        samples_a1 = dataset[dataset['fluency'] == a1]
        samples_a2 = dataset[dataset['fluency'] == a2]

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
        samples_a1 = dataset[dataset['fluency'] == a1]
        samples_a2 = dataset[dataset['fluency'] == a2]

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

    original_score_counts = count_utterances_by_score(new_ds, score_keys=["fluency"])
    print("Original score counts:", dict(original_score_counts))

    # Reformatting and mapping scores
    new_ds = new_ds.map(reformat_and_map, with_indices=True)

    # Filter out samples with None fluency
    new_ds = new_ds.filter(lambda x: x["fluency"] is not None)

    # Filter out samples with audio shorter than 0.02 seconds
    new_ds = new_ds.filter(filter_long_audio)

    # Count utterances by score in the original dataset
    original_score_counts = count_utterances_by_score(new_ds, score_keys=["fluency"])
    print("Original score counts:", dict(original_score_counts))

    # Get balanced pairs
    balanced_pairs, selected_pair_counts = get_balanced_pairs_and_counts(new_ds, target_pairs_per_pair=30)
    print(selected_pair_counts)

    # Create a new dataset with the selected pairs
    paired_data = []
    for sample1, sample2 in balanced_pairs:
        score1 = sample1["fluency"]
        score2 = sample2["fluency"]
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
    # paired_ds.push_to_hub(repo_id="DynamicSuperb/L2EnglishFluency_speechocean762-BinaryAccuracy", split="test", token=os.environ["HF_TOKEN"])
