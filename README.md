The repository is based on the public repository: [[linguistic-superb]](https://github.com/juice500ml/linguistic-superb). It includes data preparation codes for a set of tasks that are based on various schools of linguistics. The tasks include, but are not limited to, nonce word detection, phone feature classification, phone/phoneme counting, pronunciation similarity, Part of Speech (PoS) tagging, prosody naturalness, semantic textual similarity, and sentence grammar acceptability.

## L2 English Pronunciation scoring
This repository provides data preparation codes for L2 English pronunciation scoring tasks in [[Dynamic Superb]](https://github.com/dynamic-superb/dynamic-superb). The task utilizes subdatasets of [[speechocean762 dataset]](https://github.com/jimbozhang/speechocean762), to meet the criteria of Dyanimc Superb (< 1 hour). The task includes three assessment tasks, which includes accuracy scoring, fluency scoring, and prosodic aspect scoring. Furthermore, we include two subtasks for each assessment task as follows: 
   - Scoring task: correlation between scores from provided human experts and predicted results
   - Ranking task: ask the model to compare the scores of the two audios (e.g. which audio has higher scores?

## References
[1] https://github.com/juice500ml/linguistic-superb

[2] Zhang, Junbo and Zhang, Zhiwen and Wang, Yongqing and Yan, Zhiyong and Song, Qiong and Huang, Yukai and Li, Ke and Povey, Daniel and Wang, Yujun (2021). speechocean762: An open-source non-native english speech corpus for pronunciation assessment. In *proc. Interspeech 2021*
