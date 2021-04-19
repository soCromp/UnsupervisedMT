# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

set -e

#
# Data preprocessing configuration
#

N_MONO=5000000  # number of monolingual sentences for each language
CODES=50000      # number of BPE codes
N_THREADS=48     # number of threads in data preprocessing
N_EPOCHS=0      # number of fastText epochs


#
# Initialize tools and data paths
#

# main paths
UMT_PATH=$PWD
TOOLS_PATH=$PWD/tools
DATA_PATH=$PWD/data
MONO_PATH=$DATA_PATH/mono
PARA_PATH=$DATA_PATH/para
WU_PATH=$DATA_PATH/disc

# create paths
mkdir -p $TOOLS_PATH
mkdir -p $DATA_PATH
mkdir -p $MONO_PATH
mkdir -p $PARA_PATH
mkdir -p $WU_PATH

# moses
MOSES=$TOOLS_PATH/mosesdecoder
TOKENIZER=$MOSES/scripts/tokenizer/tokenizer.perl
NORM_PUNC=$MOSES/scripts/tokenizer/normalize-punctuation.perl
INPUT_FROM_SGM=$MOSES/scripts/ems/support/input-from-sgm.perl
REM_NON_PRINT_CHAR=$MOSES/scripts/tokenizer/remove-non-printing-char.perl

# fastBPE
FASTBPE_DIR=$TOOLS_PATH/fastBPE
FASTBPE=$FASTBPE_DIR/fast

# fastText
FASTTEXT_DIR=$TOOLS_PATH/fastText
FASTTEXT=$FASTTEXT_DIR/fasttext

# files full paths
SRC_RAW=$MONO_PATH/all.en
TGT_RAW=$MONO_PATH/all.kk
SRC_TOK=$MONO_PATH/all.en.tok
TGT_TOK=$MONO_PATH/all.kk.tok
BPE_CODES=$MONO_PATH/bpe_codes
CONCAT_BPE=$MONO_PATH/all.en-kk.$CODES
SRC_VOCAB=$MONO_PATH/vocab.en.$CODES
TGT_VOCAB=$MONO_PATH/vocab.kk.$CODES
FULL_VOCAB=$MONO_PATH/vocab.en-kk.$CODES
SRC_VALID=$PARA_PATH/sgm/newstest2019-kken-ref.en
TGT_VALID=$PARA_PATH/sgm/newstest2019-kken-ref.kk
SRC_TEST=$PARA_PATH/sgm/newstest2019-enkk-src.en
TGT_TEST=$PARA_PATH/sgm/newstest2019-enkk-src.kk

WU_TRAIN_SRC_RAW=$PARA_PATH/dev/newsdev2019-kken-ref.en
WU_TRAIN_TGT_RAW=$PARA_PATH/dev/newsdev2019-kken-ref.kk
WU_VALID_SRC_RAW=$PARA_PATH/sgm/newstest2019-kken-ref.en
WU_VALID_TGT_RAW=$PARA_PATH/sgm/newstest2019-kken-ref.kk
WU_TEST_SRC_RAW=$PARA_PATH/sgm/newstest2019-enkk-src.en
WU_TEST_TGT_RAW=$PARA_PATH/sgm/newstest2019-enkk-src.kk

WU_TRAIN_SRC=$WU_PATH/train.en
WU_TRAIN_TGT=$WU_PATH/train.kk
WU_VALID_SRC=$WU_PATH/valid.en
WU_VALID_TGT=$WU_PATH/valid.kk
WU_TEST_SRC=$WU_PATH/test.en
WU_TEST_TGT=$WU_PATH/test.kk

#
# Download monolingual data
#

cd $MONO_PATH

echo "Downloading English files..."
wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2007.en.shuffled.gz
wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2008.en.shuffled.gz

echo "Downloading Kazakh files..."
if ! [[ -f "$MONO_PATH/news.2019.kk.shuffled.gz" ]]; then gsutil cp gs://data1678/news.2019.kk.shuffled.gz $MONO_PATH; fi
# wget -c http://data.statmt.org/wmt19/translation-task/wiki/wiki.2018.kk.filtered.gz

# decompress monolingual data
for FILENAME in news*gz; do
  OUTPUT="${FILENAME::-3}"
  if [ ! -f "$OUTPUT" ]; then
    echo "Decompressing $FILENAME..."
    gunzip -k $FILENAME
  else
    echo "$OUTPUT already decompressed."
  fi
done

# concatenate monolingual data files
if ! [[ -f "$SRC_RAW" && -f "$TGT_RAW" ]]; then
  echo "Concatenating monolingual data..."
  cat $(ls news*en* | grep -v gz) | head -n $N_MONO > $SRC_RAW
  cat $(ls news*kk* | grep -v gz) | head -n $N_MONO > $TGT_RAW
fi
echo "EN monolingual data concatenated in: $SRC_RAW"
echo "KK monolingual data concatenated in: $TGT_RAW"

# check number of lines
if ! [[ "$(wc -l < $SRC_RAW)" -eq "$N_MONO" ]]; then echo "ERROR: Number of lines doesn't match! Be sure you have $N_MONO sentences in your EN monolingual data."; exit; fi
if ! [[ "$(wc -l < $TGT_RAW)" -eq "$N_MONO" ]]; then echo "ERROR: Number of lines doesn't match! Be sure you have $N_MONO sentences in your KK monolingual data."; exit; fi

# tokenize data
if ! [[ -f "$SRC_TOK" && -f "$TGT_TOK" ]]; then
  echo "Tokenize monolingual data..."
  cat $SRC_RAW | $NORM_PUNC -l en | $TOKENIZER -l en -no-escape -threads $N_THREADS > $SRC_TOK
  cat $TGT_RAW | $NORM_PUNC -l ru | $TOKENIZER -l ru -no-escape -threads $N_THREADS > $TGT_TOK
fi
echo "EN monolingual data tokenized in: $SRC_TOK"
echo "KK monolingual data tokenized in: $TGT_TOK"

# learn BPE codes
if [ ! -f "$BPE_CODES" ]; then
  echo "Learning BPE codes..."
  $FASTBPE learnbpe $CODES $SRC_TOK $TGT_TOK > $BPE_CODES
fi
echo "BPE learned in $BPE_CODES"

# apply BPE codes
if ! [[ -f "$SRC_TOK.$CODES" && -f "$TGT_TOK.$CODES" ]]; then
  echo "Applying BPE codes..."
  $FASTBPE applybpe $SRC_TOK.$CODES $SRC_TOK $BPE_CODES
  $FASTBPE applybpe $TGT_TOK.$CODES $TGT_TOK $BPE_CODES
fi
echo "BPE codes applied to EN in: $SRC_TOK.$CODES"
echo "BPE codes applied to KK in: $TGT_TOK.$CODES"

# extract vocabulary
if ! [[ -f "$SRC_VOCAB" && -f "$TGT_VOCAB" && -f "$FULL_VOCAB" ]]; then
  echo "Extracting vocabulary..."
  $FASTBPE getvocab $SRC_TOK.$CODES > $SRC_VOCAB
  $FASTBPE getvocab $TGT_TOK.$CODES > $TGT_VOCAB
  $FASTBPE getvocab $SRC_TOK.$CODES $TGT_TOK.$CODES > $FULL_VOCAB
fi
echo "EN vocab in: $SRC_VOCAB"
echo "KK vocab in: $TGT_VOCAB"
echo "Full vocab in: $FULL_VOCAB"

# binarize data
if ! [[ -f "$SRC_TOK.$CODES.pth" && -f "$TGT_TOK.$CODES.pth" ]]; then
  echo "Binarizing data..."
  $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_TOK.$CODES
  $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_TOK.$CODES
fi
echo "EN binarized data in: $SRC_TOK.$CODES.pth"
echo "KK binarized data in: $TGT_TOK.$CODES.pth"


#
# Download parallel data (for evaluation only)
#

cd $PARA_PATH

echo "Downloading parallel data..."
wget -c http://data.statmt.org/wmt19/translation-task/dev.tgz
wget -c http://data.statmt.org/wmt19/translation-task/test.tgz

echo "Extracting parallel data..."
tar -xzf dev.tgz
tar -xzf test.tgz

# check wu and valid and test files are here
if ! [[ -f "$SRC_VALID.sgm" ]]; then echo "$SRC_VALID.sgm is not found!"; exit; fi
if ! [[ -f "$TGT_VALID.sgm" ]]; then echo "$TGT_VALID.sgm is not found!"; exit; fi
if ! [[ -f "$SRC_TEST.sgm" ]]; then echo "$SRC_TEST.sgm is not found!"; exit; fi
if ! [[ -f "$TGT_TEST.sgm" ]]; then echo "$TGT_TEST.sgm is not found!"; exit; fi
if ! [[ -f "$WU_TRAIN_SRC_RAW.sgm" ]]; then echo "$WU_TRAIN_SRC_RAW.sgm is not found!"; exit; fi
if ! [[ -f "$WU_TRAIN_TGT_RAW.sgm" ]]; then echo "$WU_TRAIN_TGT_RAW.sgm is not found!"; exit; fi
if ! [[ -f "$WU_VALID_SRC_RAW.sgm" ]]; then echo "$WU_VALID_SRC_RAW.sgm is not found!"; exit; fi
if ! [[ -f "$WU_VALID_TGT_RAW.sgm" ]]; then echo "$WU_VALID_TGT_RAW.sgm is not found!"; exit; fi
if ! [[ -f "$WU_TEST_SRC_RAW.sgm" ]]; then echo "$WU_TEST_SRC_RAW.sgm is not found!"; exit; fi
if ! [[ -f "$WU_TEST_TGT_RAW.sgm" ]]; then echo "$WU_TEST_TGT_RAW.sgm is not found!"; exit; fi

echo "Tokenizing wu, valid and test data..."
$INPUT_FROM_SGM < $SRC_VALID.sgm | $NORM_PUNC -l en | $REM_NON_PRINT_CHAR | $TOKENIZER -l en -no-escape -threads $N_THREADS > $SRC_VALID
$INPUT_FROM_SGM < $TGT_VALID.sgm | $NORM_PUNC -l ru | $REM_NON_PRINT_CHAR | $TOKENIZER -l ru -no-escape -threads $N_THREADS > $TGT_VALID
$INPUT_FROM_SGM < $SRC_TEST.sgm | $NORM_PUNC -l en | $REM_NON_PRINT_CHAR | $TOKENIZER -l en -no-escape -threads $N_THREADS > $SRC_TEST
$INPUT_FROM_SGM < $TGT_TEST.sgm | $NORM_PUNC -l ru | $REM_NON_PRINT_CHAR | $TOKENIZER -l ru -no-escape -threads $N_THREADS > $TGT_TEST
$INPUT_FROM_SGM < $WU_TRAIN_SRC_RAW.sgm | $NORM_PUNC -l en | $REM_NON_PRINT_CHAR | $TOKENIZER -l en -no-escape -threads $N_THREADS > $WU_TRAIN_SRC_RAW
$INPUT_FROM_SGM < $WU_TRAIN_TGT_RAW.sgm | $NORM_PUNC -l ru | $REM_NON_PRINT_CHAR | $TOKENIZER -l ru -no-escape -threads $N_THREADS > $WU_TRAIN_TGT_RAW
$INPUT_FROM_SGM < $WU_VALID_SRC_RAW.sgm | $NORM_PUNC -l en | $REM_NON_PRINT_CHAR | $TOKENIZER -l en -no-escape -threads $N_THREADS > $WU_VALID_SRC_RAW
$INPUT_FROM_SGM < $WU_VALID_TGT_RAW.sgm | $NORM_PUNC -l ru | $REM_NON_PRINT_CHAR | $TOKENIZER -l ru -no-escape -threads $N_THREADS > $WU_VALID_TGT_RAW
$INPUT_FROM_SGM < $WU_TEST_SRC_RAW.sgm | $NORM_PUNC -l en | $REM_NON_PRINT_CHAR | $TOKENIZER -l en -no-escape -threads $N_THREADS > $WU_TEST_SRC_RAW
$INPUT_FROM_SGM < $WU_TEST_TGT_RAW.sgm | $NORM_PUNC -l ru | $REM_NON_PRINT_CHAR | $TOKENIZER -l ru -no-escape -threads $N_THREADS > $WU_TEST_TGT_RAW

echo "Applying BPE to wu, valid and test files..."
$FASTBPE applybpe $SRC_VALID.$CODES $SRC_VALID $BPE_CODES $SRC_VOCAB
$FASTBPE applybpe $TGT_VALID.$CODES $TGT_VALID $BPE_CODES $TGT_VOCAB
$FASTBPE applybpe $SRC_TEST.$CODES $SRC_TEST $BPE_CODES $SRC_VOCAB
$FASTBPE applybpe $TGT_TEST.$CODES $TGT_TEST $BPE_CODES $TGT_VOCAB
$FASTBPE applybpe $WU_TRAIN_SRC $WU_TRAIN_SRC_RAW $BPE_CODES $SRC_VOCAB
$FASTBPE applybpe $WU_TRAIN_TGT $WU_TRAIN_TGT_RAW $BPE_CODES $TGT_VOCAB
$FASTBPE applybpe $WU_VALID_SRC $WU_VALID_SRC_RAW $BPE_CODES $SRC_VOCAB
$FASTBPE applybpe $WU_VALID_TGT $WU_VALID_TGT_RAW $BPE_CODES $TGT_VOCAB
$FASTBPE applybpe $WU_TEST_SRC $WU_TEST_SRC_RAW $BPE_CODES $SRC_VOCAB
$FASTBPE applybpe $WU_TEST_TGT $WU_TEST_TGT_RAW $BPE_CODES $TGT_VOCAB

echo "Binarizing data..."
rm -f $SRC_VALID.$CODES.pth $TGT_VALID.$CODES.pth $SRC_TEST.$CODES.pth $TGT_TEST.$CODES.pth
$UMT_PATH/preprocess.py $FULL_VOCAB $SRC_VALID.$CODES
$UMT_PATH/preprocess.py $FULL_VOCAB $TGT_VALID.$CODES
$UMT_PATH/preprocess.py $FULL_VOCAB $SRC_TEST.$CODES
$UMT_PATH/preprocess.py $FULL_VOCAB $TGT_TEST.$CODES
$UMT_PATH/preprocess.py $FULL_VOCAB $WU_TRAIN_SRC
$UMT_PATH/preprocess.py $FULL_VOCAB $WU_TRAIN_TGT
$UMT_PATH/preprocess.py $FULL_VOCAB $WU_VALID_SRC
$UMT_PATH/preprocess.py $FULL_VOCAB $WU_VALID_TGT
$UMT_PATH/preprocess.py $FULL_VOCAB $WU_TEST_SRC
$UMT_PATH/preprocess.py $FULL_VOCAB $WU_TEST_TGT


#
# Summary
#
echo ""
echo "===== Data summary"
echo "=== Lample"
echo "Monolingual training data:"
echo "    EN: $SRC_TOK.$CODES.pth"
echo "    KK: $TGT_TOK.$CODES.pth"
echo "Parallel validation data:"
echo "    EN: $SRC_VALID.$CODES.pth"
echo "    KK: $TGT_VALID.$CODES.pth"
echo "Parallel test data:"
echo "    EN: $SRC_TEST.$CODES.pth"
echo "    KK: $TGT_TEST.$CODES.pth"
echo "=== Wu"
echo "Train:"
echo "    EN: $WU_TRAIN_SRC.pth"
echo "    KK: $WU_TRAIN_TGT.pth"
echo "Valid:"
echo "    EN: $WU_VALID_SRC.pth"
echo "    KK: $WU_VALID_TGT.pth"
echo "Test:"
echo "    EN: $WU_TEST_SRC.pth"
echo "    KK: $WU_TEST_TGT.pth"
echo ""


#
# Train fastText on concatenated embeddings
#

if ! [[ -f "$CONCAT_BPE" ]]; then
  echo "Concatenating source and target monolingual data..."
  cat $SRC_TOK.$CODES $TGT_TOK.$CODES | shuf > $CONCAT_BPE
fi
echo "Concatenated data in: $CONCAT_BPE"

if ! [[ -f "$CONCAT_BPE.vec" ]]; then
  echo "Training fastText on $CONCAT_BPE..."
  $FASTTEXT skipgram -epoch $N_EPOCHS -minCount 0 -dim 512 -thread $N_THREADS -ws 5 -neg 10 -input $CONCAT_BPE -output $CONCAT_BPE
fi
echo "Cross-lingual embeddings in: $CONCAT_BPE.vec"
