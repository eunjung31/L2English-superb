from datasets import load_dataset
from tqdm import tqdm
import soundfile as sf

download_path = "PATH_TO_DOWNLOAD"
aspect = "Accuracy"

if __name__ == "__main__":
    ds = load_dataset("DynamicSuperb/L2English" + aspect + "_speechocean762-Scoring",
                      num_proc=64)
    new_ds = ds["test"]
    print(len(new_ds))

    for data in tqdm(new_ds):
        audio_data = data['audio']['array']  # get the audio data array
        filename = data['file'] + ".wav"
        sample_rate = data['audio']['sampling_rate']  # get the sampling rate
        sf.write(download_path + f'/{filename}', audio_data, sample_rate)  # write the data to a .wav file
