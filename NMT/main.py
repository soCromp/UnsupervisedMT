# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import time
import json
import argparse
import src.options
from collections import OrderedDict

from src.data.loader import check_all_data_params, load_data
from src.utils import bool_flag, initialize_exp
from src.model import check_mt_model_params, build_mt_model
from src.model.wu_discriminator import Discriminator as Wu_Discriminator
from src.trainer import TrainerMT
from src.evaluator import EvaluatorMT

parser = argparse.ArgumentParser(description="Driver program for CS1678 Failed Project :(")

print("Processing lample args")
#lample - load args
options.add_parse_parameters_args(parser)
options.add_autoencoder_parameters_args(parser)
options.add_encoder_input_perturbation_args(parser)
options.add_discriminator_parameters_args(parser)
options.add_dataset_args(parser)
options.add_training_steps_args(parser)
options.add_language_model_training_args(parser)
options.add_training_parameters_args(parser)
options.add_reload_models_args(parser)
options.add_freeze_network_paramters_args(parser)
options.add_evaluation_args(parser)

print("Processing wu args")
# Wu - Load args
options.add_general_args(parser)
options.add_dataset_args(parser)
options.add_distributed_training_args(parser)
options.add_optimization_args(parser)
options.add_checkpoint_args(parser)
options.add_generator_model_args(parser)
options.add_discriminator_model_args(parser)
options.add_generation_args(parser)

print("Done processing args")

def train_g(params, encoder, decoder, discriminator, lm, trainer, evaluator):
    # check checkpoints saving path
    if not os.path.exists('checkpoints/generator'):
        os.makedirs('checkpoints/generator')

    checkpoints_path = 'checkpoints/generator/'

    # evaluation mode
    if params.eval_only:
        evaluator.run_all_evals(0)
        exit()

    # language model pretraining
    if params.lm_before > 0:
        logger.info("Pretraining language model for %i iterations ..." % params.lm_before)
        trainer.n_sentences = 0
        for _ in range(params.lm_before):
            for lang in params.langs:
                trainer.lm_step(lang)
            trainer.iter()

    # define epoch size
    if params.epoch_size == -1:
        params.epoch_size = params.n_para
    assert params.epoch_size > 0

    # start training
    for _ in range(trainer.epoch, params.max_epoch):

        logger.info("====================== Starting epoch %i ... ======================" % trainer.epoch)

        trainer.n_sentences = 0

        while trainer.n_sentences < params.epoch_size:

            # mono_discriminator training
            for _ in range(params.n_dis):
                trainer.discriminator_step()

            # language model training
            if params.lambda_lm > 0:
                for _ in range(params.lm_after):
                    for lang in params.langs:
                        trainer.lm_step(lang)

            # MT training (parallel data)
            if params.lambda_xe_para > 0:
                for lang1, lang2 in params.para_directions:
                    trainer.enc_dec_step(lang1, lang2, params.lambda_xe_para)

            # MT training (back-parallel data)
            if params.lambda_xe_back > 0:
                for lang1, lang2 in params.back_directions:
                    trainer.enc_dec_step(lang1, lang2, params.lambda_xe_back, back=True)

            # autoencoder training (monolingual data)
            if params.lambda_xe_mono > 0:
                for lang in params.mono_directions:
                    trainer.enc_dec_step(lang, lang, params.lambda_xe_mono)

            # AE - MT training (on the fly back-translation)
            if params.lambda_xe_otfd > 0 or params.lambda_xe_otfa > 0:

                # start on-the-fly batch generations
                if not getattr(params, 'started_otf_batch_gen', False):
                    otf_iterator = trainer.otf_bt_gen_async()
                    params.started_otf_batch_gen = True

                # update model parameters on subprocesses
                if trainer.n_iter % params.otf_sync_params_every == 0:
                    trainer.otf_sync_params()

                # get training batch from CPU
                before_gen = time.time()
                batches = next(otf_iterator)
                trainer.gen_time += time.time() - before_gen

                # training
                for batch in batches:
                    lang1, lang2, lang3 = batch['lang1'], batch['lang2'], batch['lang3']
                    # 2-lang back-translation - autoencoding
                    if lang1 != lang2 == lang3:
                        trainer.otf_bt(batch, params.lambda_xe_otfa, params.otf_backprop_temperature)
                    # 2-lang back-translation - parallel data
                    elif lang1 == lang3 != lang2:
                        trainer.otf_bt(batch, params.lambda_xe_otfd, params.otf_backprop_temperature)
                    # 3-lang back-translation - parallel data
                    elif lang1 != lang2 and lang2 != lang3 and lang1 != lang3:
                        trainer.otf_bt(batch, params.lambda_xe_otfd, params.otf_backprop_temperature)

            trainer.iter()

        # end of epoch
        logger.info("====================== End of epoch %i ======================" % trainer.epoch)

        # evaluate discriminator / perplexity / BLEU
        scores = evaluator.run_all_evals(trainer.epoch)

        # print / JSON log
        for k, v in scores.items():
            logger.info('%s -> %.6f' % (k, v))
        logger.info("__log__:%s" % json.dumps(scores))

        # save best / save periodic / end epoch
        trainer.save_best_model(scores)
        trainer.save_periodic()
        trainer.end_epoch(scores)
        trainer.test_sharing()

def main(params):
    # check parameters
    assert params.exp_name
    check_all_data_params(params)
    check_mt_model_params(params)

    # initialize experiment / load data / build model
    logger = initialize_exp(params)
    data = load_data(params)
    encoder, decoder, mono_discriminator, lm = build_mt_model(params, data)
    para_discriminator = Wu_Discriminator(args, data['wu'].src_dict, data['wu'].dst_dict, use_cuda=cuda)

    # initialize trainer / reload checkpoint / initialize evaluator
    trainer = TrainerMT(encoder, decoder, mono_discriminator, lm, data, params)
    trainer.reload_checkpoint()
    trainer.test_sharing()  # check parameters sharing
    evaluator = EvaluatorMT(trainer, data, params)

    train_g(params, encoder, decoder, mono_discriminator, lm, trainer, evaluator)

    train_d(params, data['wu'], generator)

    g_logging_meters = OrderedDict()
    g_logging_meters['train_loss'] = AverageMeter()
    g_logging_meters['valid_loss'] = AverageMeter()
    g_logging_meters['train_acc'] = AverageMeter()
    g_logging_meters['valid_acc'] = AverageMeter()
    g_logging_meters['bsz'] = AverageMeter()  # sentences per batch

    d_logging_meters = OrderedDict()
    d_logging_meters['train_loss'] = AverageMeter()
    d_logging_meters['valid_loss'] = AverageMeter()
    d_logging_meters['train_acc'] = AverageMeter()
    d_logging_meters['valid_acc'] = AverageMeter()
    d_logging_meters['bsz'] = AverageMeter()  # sentences per batch

    joint_train(params, dataset, generator, discriminator, g_logging_meters, d_logging_meters)


if __name__ == '__main__':

    ret = parser.parse_known_args()
    options = ret[0]
    if ret[1]:
        logging.warning("unknown arguments: {0}".format(parser.parse_known_args()[1]))
    #main(options)
    #parser = options.get_parser()
    #params = parser.parse_args()
    main(options)
