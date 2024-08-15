# For ranking task, the audio files may not be unique. Therefore, we rename the audio files into {i}_{filename}.wav format.

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

    for i, data in enumerate(new_ds):
        audio_data1 = data['audio']['array']  # get the audio data array
        audio_data2 = data['audio2']['array']  # get the audio data array
        filename1 = data['audio']['path'].split(".")[0]
        filename2 = data['audio2']['path'].split(".")[0]
        sample_rate1 = data['audio']['sampling_rate']  # get the sampling rate
        sample_rate2 = data['audio2']['sampling_rate']  # get the sampling rate

        # Convert i to string and pad with zeros to get a string of length 3
        i_str = str(i).zfill(3)

        sf.write(download_path + f'/train/{i_str}_{filename1}.wav', audio_data1, sample_rate1)  # write the data to a .wav file
        sf.write(download_path + f'/test/{i_str}_{filename2}.wav', audio_data2, sample_rate2)  # write the data to a .wav file