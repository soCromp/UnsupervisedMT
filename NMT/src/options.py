# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

import argparse
import torch

def add_parse_parameters_args(parser):
    # parse parameters
    parser = argparse.ArgumentParser(description='Language transfer')
    parser.add_argument("--exp_name", type=str, default="",
                        help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="",
                        help="Experiment ID")
    parser.add_argument("--dump_path", type=str, default="./dumped/",
                        help="Experiment dump path")
    parser.add_argument("--save_periodic", type=bool_flag, default=False,
                        help="Save the model periodically")
    parser.add_argument("--seed", type=int, default=-1,
                        help="Random generator seed (-1 for random)")

def add_autoencoder_parameters_args(parser):
    # autoencoder parameters
    parser.add_argument("--emb_dim", type=int, default=512,
                        help="Embedding layer size")
    parser.add_argument("--n_enc_layers", type=int, default=4,
                        help="Number of layers in the encoders")
    parser.add_argument("--n_dec_layers", type=int, default=4,
                        help="Number of layers in the decoders")
    parser.add_argument("--hidden_dim", type=int, default=512,
                        help="Hidden layer size")
    parser.add_argument("--lstm_proj", type=bool_flag, default=False,
                        help="Projection layer between decoder LSTM and output layer")
    parser.add_argument("--dropout", type=float, default=0,
                        help="Dropout")
    parser.add_argument("--label-smoothing", type=float, default=0,
                        help="Label smoothing")
    parser.add_argument("--attention", type=bool_flag, default=True,
                        help="Use an attention mechanism")
    if not parser.parse_known_args()[0].attention:
        parser.add_argument("--enc_dim", type=int, default=512,
                            help="Latent space dimension")
        parser.add_argument("--proj_mode", type=str, default="last",
                            help="Projection mode (proj / pool / last)")
        parser.add_argument("--init_encoded", type=bool_flag, default=False,
                            help="Initialize the decoder with the encoded state. Append it to each input embedding otherwise.")
    else:
        parser.add_argument("--transformer", type=bool_flag, default=True,
                            help="Use transformer architecture + attention mechanism")
        if parser.parse_known_args()[0].transformer:
            parser.add_argument("--transformer_ffn_emb_dim", type=int, default=2048,
                                help="Transformer fully-connected hidden dim size")
            parser.add_argument("--attention_dropout", type=float, default=0,
                                help="attention_dropout")
            parser.add_argument("--relu_dropout", type=float, default=0,
                                help="relu_dropout")
            parser.add_argument("--encoder_attention_heads", type=int, default=8,
                                help="encoder_attention_heads")
            parser.add_argument("--decoder_attention_heads", type=int, default=8,
                                help="decoder_attention_heads")
            parser.add_argument("--encoder_normalize_before", type=bool_flag, default=False,
                                help="encoder_normalize_before")
            parser.add_argument("--decoder_normalize_before", type=bool_flag, default=False,
                                help="decoder_normalize_before")
        else:
            parser.add_argument("--input_feeding", type=bool_flag, default=False,
                                help="Input feeding")
            parser.add_argument("--share_att_proj", type=bool_flag, default=False,
                                help="Share attention projetion layer")
    parser.add_argument("--share_lang_emb", type=bool_flag, default=False,
                        help="Share embedding layers between languages (enc / dec / proj)")
    parser.add_argument("--share_encdec_emb", type=bool_flag, default=False,
                        help="Share encoder embeddings / decoder embeddings")
    parser.add_argument("--share_decpro_emb", type=bool_flag, default=False,
                        help="Share decoder embeddings / decoder output projection")
    parser.add_argument("--share_output_emb", type=bool_flag, default=False,
                        help="Share decoder output embeddings")
    parser.add_argument("--share_lstm_proj", type=bool_flag, default=False,
                        help="Share projection layer between decoder LSTM and output layer)")
    parser.add_argument("--share_enc", type=int, default=0,
                        help="Number of layers to share in the encoders")
    parser.add_argument("--share_dec", type=int, default=0,
                        help="Number of layers to share in the decoders")

def add_encoder_input_perturbation_args(parser):
    # encoder input perturbation
    parser.add_argument("--word_shuffle", type=float, default=0,
                        help="Randomly shuffle input words (0 to disable)")
    parser.add_argument("--word_dropout", type=float, default=0,
                        help="Randomly dropout input words (0 to disable)")
    parser.add_argument("--word_blank", type=float, default=0,
                        help="Randomly blank input words (0 to disable)")

def add_discriminator_parameters_args(parser):
    # discriminator parameters
    parser.add_argument("--dis_layers", type=int, default=3,
                        help="Number of hidden layers in the discriminator")
    parser.add_argument("--dis_hidden_dim", type=int, default=128,
                        help="Discriminator hidden layers dimension")
    parser.add_argument("--dis_dropout", type=float, default=0,
                        help="Discriminator dropout")
    parser.add_argument("--dis_clip", type=float, default=0,
                        help="Clip discriminator weights (0 to disable)")
    parser.add_argument("--dis_smooth", type=float, default=0,
                        help="GAN smooth predictions")
    parser.add_argument("--dis_input_proj", type=bool_flag, default=True,
                        help="Feed the discriminator with the projected output (attention only)")

def add_dataset_args(parser):
    # dataset
    parser.add_argument("--langs", type=str, default="",
                        help="Languages (lang1,lang2)")
    parser.add_argument("--vocab", type=str, default="",
                        help="Vocabulary (lang1:path1;lang2:path2)")
    parser.add_argument("--vocab_min_count", type=int, default=0,
                        help="Vocabulary minimum word count")
    parser.add_argument("--mono_dataset", type=str, default="",
                        help="Monolingual dataset (lang1:train1,valid1,test1;lang2:train2,valid2,test2)")
    parser.add_argument("--para_dataset", type=str, default="",
                        help="Parallel dataset (lang1-lang2:train12,valid12,test12;lang1-lang3:train13,valid13,test13)")
    parser.add_argument("--back_dataset", type=str, default="",
                        help="Back-parallel dataset, with noisy source and clean target (lang1-lang2:train121,train122;lang2-lang1:train212,train211)")
    parser.add_argument("--n_mono", type=int, default=0,
                        help="Number of monolingual sentences (-1 for everything)")
    parser.add_argument("--n_para", type=int, default=0,
                        help="Number of parallel sentences (-1 for everything)")
    parser.add_argument("--n_back", type=int, default=0,
                        help="Number of back-parallel sentences (-1 for everything)")
    parser.add_argument("--max_len", type=int, default=175,
                        help="Maximum length of sentences (after BPE)")
    parser.add_argument("--max_vocab", type=int, default=-1,
                        help="Maximum vocabulary size (-1 to disable)")
    
def add_training_steps_args(parser):
    # training steps
    parser.add_argument("--n_dis", type=int, default=0,
                        help="Number of discriminator training iterations")
    parser.add_argument("--mono_directions", type=str, default="",
                        help="Training directions (lang1,lang2)")
    parser.add_argument("--para_directions", type=str, default="",
                        help="Training directions (lang1-lang2,lang2-lang1)")
    parser.add_argument("--pivo_directions", type=str, default="",
                        help="Training directions with online back-translation, using a pivot (lang1-lang3-lang1,lang1-lang3-lang2)]")
    parser.add_argument("--back_directions", type=str, default="",
                        help="Training directions with back-translation dataset (lang1-lang2)")
    parser.add_argument("--otf_sample", type=float, default=-1,
                        help="Temperature for sampling back-translations (-1 for greedy decoding)")
    parser.add_argument("--otf_backprop_temperature", type=float, default=-1,
                        help="Back-propagate through the encoder (-1 to disable, temperature otherwise)")
    parser.add_argument("--otf_sync_params_every", type=int, default=1000, metavar="N",
                        help="Number of updates between synchronizing params")
    parser.add_argument("--otf_num_processes", type=int, default=30, metavar="N",
                        help="Number of processes to use for OTF generation")
    parser.add_argument("--otf_update_enc", type=bool_flag, default=True,
                        help="Update the encoder during back-translation training")
    parser.add_argument("--otf_update_dec", type=bool_flag, default=True,
                        help="Update the decoder during back-translation training")

def add_language_model_training_args(parser):
    # language model training
    parser.add_argument("--lm_before", type=int, default=0,
                        help="Training steps with language model pretraining (0 to disable)")
    parser.add_argument("--lm_after", type=int, default=0,
                        help="Keep training the language model during MT training (0 to disable)")
    parser.add_argument("--lm_share_enc", type=int, default=0,
                        help="Number of shared LSTM layers in the encoder")
    parser.add_argument("--lm_share_dec", type=int, default=0,
                        help="Number of shared LSTM layers in the decoder")
    parser.add_argument("--lm_share_emb", type=bool_flag, default=False,
                        help="Share language model lookup tables")
    parser.add_argument("--lm_share_proj", type=bool_flag, default=False,
                        help="Share language model projection layers")

def add_training_parameters_args(parser):
    # training parameters
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--group_by_size", type=bool_flag, default=True,
                        help="Sort sentences by size during the training")
    parser.add_argument("--lambda_xe_mono", type=str, default="0",
                        help="Cross-entropy reconstruction coefficient (autoencoding)")
    parser.add_argument("--lambda_xe_para", type=str, default="0",
                        help="Cross-entropy reconstruction coefficient (parallel data)")
    parser.add_argument("--lambda_xe_back", type=str, default="0",
                        help="Cross-entropy reconstruction coefficient (back-parallel data)")
    parser.add_argument("--lambda_xe_otfd", type=str, default="0",
                        help="Cross-entropy reconstruction coefficient (on-the-fly back-translation parallel data)")
    parser.add_argument("--lambda_xe_otfa", type=str, default="0",
                        help="Cross-entropy reconstruction coefficient (on-the-fly back-translation autoencoding data)")
    parser.add_argument("--lambda_dis", type=str, default="0",
                        help="Discriminator loss coefficient")
    parser.add_argument("--lambda_lm", type=str, default="0",
                        help="Language model loss coefficient")
    parser.add_argument("--enc_optimizer", type=str, default="adam,lr=0.0003",
                        help="Encoder optimizer (SGD / RMSprop / Adam, etc.)")
    parser.add_argument("--dec_optimizer", type=str, default="enc_optimizer",
                        help="Decoder optimizer (SGD / RMSprop / Adam, etc.)")
    parser.add_argument("--dis_optimizer", type=str, default="rmsprop,lr=0.0005",
                        help="Discriminator optimizer (SGD / RMSprop / Adam, etc.)")
    parser.add_argument("--clip_grad_norm", type=float, default=5,
                        help="Clip gradients norm (0 to disable)")
    parser.add_argument("--epoch_size", type=int, default=100000,
                        help="Epoch size / evaluation frequency")
    parser.add_argument("--stopping_criterion", type=str, default="",
                        help="Stopping criterion, and number of non-increase before stopping the experiment")
    
def add_reload_models_args(parser):
    # reload models
    parser.add_argument("--pretrained_emb", type=str, default="",
                        help="Reload pre-trained source and target word embeddings")
    parser.add_argument("--pretrained_out", type=bool_flag, default=False,
                        help="Pretrain the decoder output projection matrix")
    parser.add_argument("--reload_model", type=str, default="",
                        help="Reload a pre-trained model")
    parser.add_argument("--reload_enc", type=bool_flag, default=False,
                        help="Reload a pre-trained encoder")
    parser.add_argument("--reload_dec", type=bool_flag, default=False,
                        help="Reload a pre-trained decoder")
    parser.add_argument("--reload_dis", type=bool_flag, default=False,
                        help="Reload a pre-trained discriminator")
    parser.add_argument("--reload_dis", type=bool_flag, default=False,
                        help="Reload a pre-trained discriminator")
    
def add_freeze_network_paramters_args(parser):
    # freeze network parameters
    parser.add_argument("--freeze_enc_emb", type=bool_flag, default=False,
                        help="Freeze encoder embeddings")
    parser.add_argument("--freeze_dec_emb", type=bool_flag, default=False,
                        help="Freeze decoder embeddings")

def add_evaluation_args(parser):
    # evaluation
    parser.add_argument("--eval_only", type=bool_flag, default=False,
                        help="Only run evaluations")
    parser.add_argument("--beam_size", type=int, default=0,
                        help="Beam width (<= 0 means greedy)")
    parser.add_argument("--length_penalty", type=float, default=1.0,
                        help="Length penalty: <1.0 favors shorter, >1.0 favors longer sentences")
    return parser

#################################################################################################
#                                       Wu Options                                              #
#################################################################################################

def add_general_args(parser):
    parser.add_argument("--wu_seed", default=1, type=int,
                        help="Random seed. (default=1)")
    return parser

def add_dataset_args(parser):
    parser.add_argument("--wu_data", required=True,
                        help="File prefix for training set.")
    parser.add_argument("--src_lang", default="de",
                        help="Source Language. (default = de)")
    parser.add_argument("--trg_lang", default="en",
                        help="Target Language. (default = en)")
    parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                       help='max number of tokens in the source sequence')
    parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                       help='max number of tokens in the target sequence')
    parser.add_argument('--skip-invalid-size-inputs-valid-test', default=True, type=bool,
                       help='Ignore too long or too short lines in valid and test set')
    parser.add_argument('--max-tokens', default=6000, type=int, metavar='N',
                       help='maximum number of tokens in a batch')
    parser.add_argument('--max-sentences', '--batch-size', type=int, metavar='N',
                       help='maximum number of sentences in a batch')
    parser.add_argument('--joint-batch-size', type=int, default=32, metavar='N',
                        help='batch size for joint training')
    parser.add_argument('--prepare-dis-batch-size', type=int, default=128, metavar='N',
                        help='batch size for preparing discriminator training')

    return parser

def add_distributed_training_args(parser):
    parser.add_argument('--distributed-world-size', type=int, metavar='N',
                       default=torch.cuda.device_count(),
                       help='total number of GPUs across all nodes (default: all visible GPUs)')
    parser.add_argument('--distributed-rank', default=0, type=int,
                       help='rank of the current worker')
    parser.add_argument("--gpuid", default=[], nargs='+', type=int,
                        help="ID of gpu device to use. Empty implies cpu usage.")

    return parser

def add_optimization_args(parser):
    parser.add_argument('--max-epoch', '--me', default=0, type=int, metavar='N',
                        help='force stop training at specified epoch')
    parser.add_argument("--epochs", default=12, type=int,
                        help="Epochs through the data. (default=12)")
    parser.add_argument("--optimizer", default="Adam", choices=["SGD", "Adadelta", "Adam"],
                        help="Optimizer of choice for training. (default=Adam)")
    parser.add_argument("--g_optimizer", default="SGD", choices=["SGD", "Adadelta", "Adam"],
                        help="Optimizer of choice for training. (default=Adam)")
    parser.add_argument("--d_optimizer", default="SGD", choices=["SGD", "Adadelta", "Adam"],
                        help="Optimizer of choice for training. (default=Adam)")
    parser.add_argument("--learning_rate", "-lr", default=1e-3, type=float,
                        help="Learning rate of the optimization. (default=0.1)")
    parser.add_argument("--g_learning_rate", "-glr", default=1e-3, type=float,
                        help="Learning rate of the generator. (default=0.001)")
    parser.add_argument("--d_learning_rate", "-dlr", default=1e-3, type=float,
                        help="Learning rate of the discriminator. (default=0.001)")
    parser.add_argument("--lr_shrink", default=0.5, type=float,
                        help='learning rate shrink factor, lr_new = (lr * lr_shrink)')
    parser.add_argument('--min-g-lr', default=1e-5, type=float, metavar='LR',
                        help='minimum learning rate')
    parser.add_argument('--min-d-lr', default=1e-6, type=float, metavar='LR',
                        help='minimum learning rate')
    parser.add_argument("--momentum", default=0.9, type=float,
                        help="Momentum when performing SGD. (default=0.9)")
    parser.add_argument("--use_estop", default=False, type=bool,
                        help="Whether use early stopping criteria. (default=False)")
    parser.add_argument("--estop", default=1e-2, type=float,
                        help="Early stopping criteria on the development set. (default=1e-2)")
    parser.add_argument('--clip-norm', default=5.0, type=float,
                       help='clip threshold of gradients')
    parser.add_argument('--curriculum', default=0, type=int, metavar='N',
                       help='sort batches by source length for first N epochs')
    parser.add_argument('--sample-without-replacement', default=0, type=int, metavar='N',
                       help='If bigger than 0, use that number of mini-batches for each epoch,'
                            ' where each sample is drawn randomly without replacement from the'
                            ' dataset')
    parser.add_argument('--sentence-avg', action='store_true',
                       help='normalize gradients by the number of sentences in a batch'
                            ' (default is to normalize by number of tokens)')

    return parser


def add_checkpoint_args(parser):
    parser.add_argument("--model_file", help="Location to dump the models.")
    return parser

def add_generator_model_args(parser):
    parser.add_argument('--encoder-embed-dim', default=512, type=int,
                       help='encoder embedding dimension')
    parser.add_argument('--encoder-layers', default=1, type=int,
                       help='encoder layers [(dim, kernel_size), ...]')
    parser.add_argument('--decoder-embed-dim', default=512, type=int,
                       help='decoder embedding dimension')
    parser.add_argument('--decoder-layers', default=1, type=int,
                       help='decoder layers [(dim, kernel_size), ...]')
    parser.add_argument('--decoder-out-embed-dim', default=512, type=int,
                       help='decoder output dimension')
    parser.add_argument('--encoder-dropout-in', default=0.1, type=float,
                       help='dropout probability for encoder input embedding')
    parser.add_argument('--encoder-dropout-out', default=0.1, type=float,
                       help='dropout probability for encoder output')
    parser.add_argument('--decoder-dropout-in', default=0.1, type=float,
                       help='dropout probability for decoder input embedding')
    parser.add_argument('--decoder-dropout-out', default=0.1, type=float,
                       help='dropout probability for decoder output')
    parser.add_argument('--wu_dropout', default=0.1, type=float, metavar='D',
                        help='dropout probability')
    parser.add_argument('--bidirectional', default=False, type=bool,
                       help='unidirectional or bidirectional encoder')
    return parser

def add_discriminator_model_args(parser):
    parser.add_argument('--fixed-max-len', default=50, type=int,
                       help='the max length the discriminator can hold')
    parser.add_argument('--d-sample-size', default=5000, type=int,
                       help='how many data used to pretrain d in one epoch')
    return parser

def add_generation_args(parser):
    parser.add_argument('--beam', default=5, type=int, metavar='N',
                        help='beam size')
    parser.add_argument('--nbest', default=1, type=int, metavar='N',
                        help='number of hypotheses to output')
    parser.add_argument('--max-len-a', default=0, type=float, metavar='N',
                        help=('generate sequences of maximum length ax + b, '
                              'where x is the source length'))
    parser.add_argument('--max-len-b', default=200, type=int, metavar='N',
                        help=('generate sequences of maximum length ax + b, '
                              'where x is the source length'))
    parser.add_argument('--remove-bpe', nargs='?', const='@@ ', default=None,
                        help='remove BPE tokens before scoring')
    parser.add_argument('--no-early-stop', action='store_true',
                        help=('continue searching even after finalizing k=beam '
                              'hypotheses; this is more correct, but increases '
                              'generation time by 50%%'))
    parser.add_argument('--unnormalized', action='store_true',
                        help='compare unnormalized hypothesis scores')
    parser.add_argument('--lenpen', default=1, type=float,
                        help='length penalty: <1.0 favors shorter, >1.0 favors longer sentences')
    parser.add_argument('--unkpen', default=0, type=float,
                        help='unknown word penalty: <0 produces more unks, >0 produces fewer')
    parser.add_argument('--replace-unk', nargs='?', const=True, default=None,
                        help='perform unknown replacement (optionally with alignment dictionary)')

    return parser
