import torch
from src.data.loader import load_data
from src.trainer import TrainerMT
from src.model import build_mt_model


def pred_batch(batch, model, params):
    lang1_id = ''  # todo
    lang2_id = ''
    max_len = 200

    sent1, len1 = batch
    sent1 = sent1.cuda()
    encoded = model.encoder(sent1, len1, lang1_id)
    sent2, lengths, one_hot = model.decoder.generate(encoded, lang2_id, max_len=max_len)
    return sent1, sent2


if __name__ == '__main__':
    params = []  # todo
    data = load_data(params)
    encoder, decoder, discriminator, lm = build_mt_model(params, data)
    # import model
    trainer = TrainerMT(encoder, decoder, discriminator, lm, data, params)
    trainer.reload_checkpoint()
    batch = ''  # make batch
    pred_batch(batch)
