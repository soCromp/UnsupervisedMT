#!/usr/bin/env bash

# #SBATCH --job-name=lample
# #SBATCH --output=lample.out
# #SBATCH --nodes=1
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=1
# #SBATCH --partition=gpu
# #SBATCH --cluster=epiphany
# #SBATCH --time=1-0:00:00

#get tools eg python 3, moses
# module load python/3.7.0
chmod +x get_tools.sh
./get_tools.sh

#get data
../../google-cloud-sdk/bin/gsutil cp gs://data1678/* .
tar -zxvf monoFREN.tar.gz # monolingual processed data
tar -zxvf paraFREN.tar.gz # raw data
tar -zxvf dataFREN.tar.gz
rm *.tar.gz

MONO_DATASET='en:./data/processed/en-fr/train.en.pth,,;fr:./data/processed/en-fr/train.fr.pth,,'
PARA_DATASET='en-fr:,./data/para/dev/newstest2013-ref.XX.60000.pth,./data/para/dev/newstest2014-fren-src.XX.60000.pth'
PRETRAINED='./data/mono/all.en-fr.60000.vec'

python main.py --exp_name test --transformer True --n_enc_layers 4 \
  --n_dec_layers 4 --share_enc 3 --share_dec 3 --share_lang_emb True \
  --share_output_emb True --langs 'en,fr' --n_mono -1 --mono_dataset \
  'en:./data/mono/all.en.tok.60000.pth,,;fr:./data/mono/all.fr.tok.60000.pth,,' \
  --para_dataset 'en-fr:,./data/para/dev/newstest2013-ref.XX.60000.pth,./data/para/dev/newstest2014-fren-src.XX.60000.pth' \
  --mono_directions 'en,fr' --word_shuffle 3 --word_dropout 0.1 --word_blank 0.2 \
  --pivo_directions 'fr-en-fr,en-fr-en' --pretrained_emb './data/mono/all.en-fr.60000.vec' \
  --pretrained_out True --lambda_xe_mono '0:1,100000:0.1,300000:0' --lambda_xe_otfd 1 \
  --otf_num_processes 30 --otf_sync_params_every 1000 --enc_optimizer adam,lr=0.0001 \
  --epoch_size 500000 --max_epoch 1
