#!/usr/bin/env bash

#inspried from:
	#https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Translation/Transformer/examples/translation/prepare-iwslt14.sh
	#https://github.com/pytorch/fairseq/blob/master/examples/translation/prepare-iwslt14.sh 

# run the script that downloads the dataset and tokenizes it
echo running prepare-iwslt14.sh
bash prepare-iwslt14.sh

# run the script to shorten all sentences to less than 50 words
echo running short-sentences.py
python short-sentences.py iwslt14.tokenized.de-en de-en 30

#taken from: #https://github.com/lisa-groundhog/GroundHog/tree/master/experiments/nmt/preprocess

# run the preprocessing scripts (creating lang dictionaries (-v 30000 is the max length of the list of vocab)) on the data
echo running preprocess scripts
python preprocess.py -d ../data/iwslt14.tokenized.de-en/vocab.en.pkl -v 30000 -b binarized_text.en.pkl ../data/iwslt14.tokenized.de-en/*.en
python preprocess.py -d ../data/iwslt14.tokenized.de-en/vocab.de.pkl -v 30000 -b binarized_text.de.pkl ../data/iwslt14.tokenized.de-en/*.de

# convert the pickeled vocab.*.pkl to dict.*.txt in the iwslt data folder
echo running pickle-to-dict.py
python pickle-to-dict.py