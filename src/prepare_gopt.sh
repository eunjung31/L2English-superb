kaldi_path="kaldi_path"
gopt_path="gopt_path"

export KALDI_ROOT=$kaldi_path

cd $gopt_path
mkdir -p data/raw_kaldi_gop/librispeech
cp src/extract_kaldi_gop/{extract_gop_feats.py,extract_gop_feats_word.py} ${kaldi_path}/egs/gop_speechocean762/s5/local/
cd ${kaldi_path}/egs/gop_speechocean762/s5
python local/extract_gop_feats.py
python local/extract_gop_feats_word.py
cd $gopt_path
cp -r ${kaldi_path}/egs/gop_speechocean762/s5/gopt_feats/* data/raw_kaldi_gop/librispeech

mkdir data/seq_data_librispeech
cd src/prep_data
python gen_seq_data_phn.py
python gen_seq_data_word.py
python gen_seq_data_utt.py

# cd $gopt_path/src
# sbatch run.sh
