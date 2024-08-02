from pathlib import Path
from typing import Dict

from datasets import Audio, Dataset
from tqdm import tqdm


def rows_to_dataset(rows: Dict[str, list]) -> Dataset:
    for key in ("audio", "audio2", "audio3", "audio4"):
        if key in rows:
            file_key = "file" if key == "audio" else f"file{key[-1]}"
            rows[file_key] = [Path(f).name for f in rows[key]]
            assert all([Path(f).exists() for f in rows[key]])

    ds = Dataset.from_dict(rows)
    for key in rows.keys():
        if "audio" in key:
            ds = ds.cast_column(key, Audio(sampling_rate=16000))

    return ds


def validate_dataset(ds: Dataset) -> None:
    print("Validating dataset...")

    # column name check
    assert len(set(ds.features.keys()) - {"instruction", "label", "audio", "audio2", "audio3", "audio4", "file", "file2", "file3", "file4"}) == 0

    # label check
    sample_size = len(ds["label"])
    assert sample_size >= 32
    assert all([(v != None) and (v != "") for v in ds["label"]])

    # instruction check
    instruction_size = len(set(ds["instruction"]))
    assert instruction_size >= max(sample_size / 20, 10)
    assert all([isinstance(v, str) for v in ds["instruction"]])
    assert all([len(v) > 0 for v in ds["instruction"]])

    # file uniqueness check
    num_audios = (len(ds.features) // 2) - 1
    if num_audios > 1:
        assert sample_size == len(set(zip(*[ds[f"file{'' if i == 0 else i+1}"] for i in range(num_audios)])))
    # the first file will be used as an index by DynamicSUPERB and thus needs to be unique
    assert sample_size == len(set(ds["file"]))

    # audio length check
    total_audio_length = 0
    for key in ("audio", "audio1", "audio2", "audio3", "audio4"):
        if key in ds.features:
            assert ds.features[key].sampling_rate == 16_000
            for audio in tqdm(ds[key]):
                assert len(audio["array"]) > 0
                total_audio_length += len(audio["array"])
    assert total_audio_length / 16_000 < 3600

    print("Dataset validated!")
    print(f"Sample size: {sample_size}")
    print(f"Unique instructions: {instruction_size}")
    print(f"Audio length: {total_audio_length / 16_000 / 60:.1f} minutes")
