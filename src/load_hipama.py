import torch
import sys
import os
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Add the model directory to the system path
sys.path.append(os.path.abspath('../src/'))
from models import HiPAMA

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GoPDataset(Dataset):
    def __init__(self, set, am='librispeech'):
        # Normalize the input to 0 mean and unit std.
        if am == 'librispeech':
            dir = 'seq_data_librispeech'
            norm_mean, norm_std = 3.203, 4.045
        elif am == 'paiia':
            dir = 'seq_data_paiia'
            norm_mean, norm_std = -0.652, 9.737
        elif am == 'paiib':
            dir = 'seq_data_paiib'
            norm_mean, norm_std = -0.516, 9.247
        else:
            raise ValueError('Acoustic Model Unrecognized.')

        # Load the dataset
        data_path = f'/home/eyeo2/workspace/hipama/data/{dir}'
        if set == 'train':
            self.feat = torch.tensor(np.load(f'{data_path}/tr_feat.npy'), dtype=torch.float)
            self.phn_label = torch.tensor(np.load(f'{data_path}/tr_label_phn.npy'), dtype=torch.float)
            self.utt_label = torch.tensor(np.load(f'{data_path}/tr_label_utt.npy'), dtype=torch.float)
            self.word_label = torch.tensor(np.load(f'{data_path}/tr_label_word.npy'), dtype=torch.float)
        elif set == 'test':
            self.feat = torch.tensor(np.load(f'{data_path}/te_feat.npy'), dtype=torch.float)
            self.phn_label = torch.tensor(np.load(f'{data_path}/te_label_phn.npy'), dtype=torch.float)
            self.utt_label = torch.tensor(np.load(f'{data_path}/te_label_utt.npy'), dtype=torch.float)
            self.word_label = torch.tensor(np.load(f'{data_path}/te_label_word.npy'), dtype=torch.float)

        # Normalize the GOP feature using the training set mean and std (only count the valid token features, exclude the padded tokens).
        self.feat = self.norm_valid(self.feat, norm_mean, norm_std)

        # Normalize the utt_label to 0-2 (same with phn score range)
        self.utt_label = self.utt_label / 5
        # The last dim is word_id, so not normalizing
        self.word_label[:, :, 0:3] = self.word_label[:, :, 0:3] / 5

    # Only normalize valid tokens, not padded token
    def norm_valid(self, feat, norm_mean, norm_std):
        norm_feat = torch.zeros_like(feat)
        for i in range(feat.shape[0]):
            for j in range(feat.shape[1]):
                if feat[i, j, 0] != 0:
                    norm_feat[i, j, :] = (feat[i, j, :] - norm_mean) / norm_std
                else:
                    break
        return norm_feat

    def __len__(self):
        return self.feat.shape[0]

    def __getitem__(self, idx):
        return self.feat[idx, :], self.phn_label[idx, :, 1], self.phn_label[idx, :, 0], self.utt_label[idx, :], self.word_label[idx, :]

# Load the pretrained model
hipama = HiPAMA(embed_dim=24, num_heads=1, depth=3, input_dim=84)
hipama = torch.nn.DataParallel(hipama)
sd = torch.load('/home/eyeo2/workspace/hipama/pretrained_models/hipama_librispeech/best_audio_model.pth', map_location='cpu')
hipama.load_state_dict(sd, strict=True)

# Move the model to the GPU
hipama = hipama.to(device)

# Load the train dataset
train_dataset = GoPDataset(set='train')
train_loader = DataLoader(train_dataset, batch_size=2500, shuffle=False)

# Load the test dataset
test_dataset = GoPDataset(set='test')
test_loader = DataLoader(test_dataset, batch_size=2500, shuffle=False)

def validate(audio_model, val_loader):
    audio_model.eval()
    A_u1, A_u2, A_u3, A_u4, A_u5, A_utt_target = [], [], [], [], [], []
    with torch.no_grad():
        for audio_input, phn_label, phns, utt_label, word_label in val_loader:
            audio_input = audio_input.to(device)

            # Compute output
            u1, u2, u3, u4, u5, p, w1, w2, w3 = audio_model(audio_input, phns)
            p, u1, u2, u3, u4, u5 = [x.to('cpu').detach() for x in [p, u1, u2, u3, u4, u5]]

            A_u1.append(u1)
            A_u2.append(u2)
            A_u3.append(u3)
            A_u4.append(u4)
            A_u5.append(u5)
            A_utt_target.append(utt_label)

        # Concatenate results
        A_u1, A_u2, A_u3, A_u4, A_u5, A_utt_target = torch.cat(A_u1), torch.cat(A_u2), torch.cat(A_u3), torch.cat(A_u4), torch.cat(A_u5), torch.cat(A_utt_target)

        A_utt = torch.cat((A_u1, A_u2, A_u3, A_u4, A_u5), dim=1)
        utt_corr = valid_utt(A_utt, A_utt_target)

    return A_utt, A_utt_target, utt_corr

def valid_utt(audio_output, target):
    corr = []
    for i in range(5):
        cur_corr = np.corrcoef(audio_output[:, i], target[:, i])[0, 1]
        corr.append(cur_corr)
    return corr

# Evaluate the model
## scoring
te_utt, te_utt_target, te_utt_corr = validate(gopt, test_loader)
print("Correlation:", te_utt_corr)

tr_utt, tr_utt_target, tr_utt_corr = validate(gopt, train_loader)

def calculate_ranking(A_utt, B_utt, A_utt_target, B_utt_target, task):
    scores = [0, 0, 0]
    task_to_index = {"accuracy": 0, "fluency": 2, "prosodic": 3}
    index = task_to_index[task]

    for i in range(len(A_utt)):
        if A_utt_target[i][index] < B_utt_target[i][index]:
            if A_utt[i][index] < B_utt[i][index]:
                scores[index] += 1

    return [score / len(A_utt) for score in scores], scores, len(A_utt)

## ranking
print("Ranking:", calculate_ranking(tr_utt, te_utt, tr_utt_target, te_utt_target, "accuracy"))