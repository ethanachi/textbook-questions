import torch
import torch.nn as nn

import onmt
import onmt.inputters
import onmt.modules
import onmt.utils

import yaml
import os
import logging
from argparse import ArgumentParser

onmt.utils.logging.init_logger()

def loadVocab(filename):
  vocab_fields = torch.load(filename)
  src_text_field = vocab_fields["src"].base_field
  tgt_text_field = vocab_fields["tgt"].base_field
  src_vocab = src_text_field.vocab
  tgt_vocab = tgt_text_field.vocab
  tgt_text_field = vocab_fields['tgt'].base_field
  src_padding = src_vocab.stoi[src_text_field.pad_token]
  tgt_padding = tgt_vocab.stoi[tgt_text_field.pad_token]
  return vocab_fields, src_text_field, tgt_text_field, src_vocab, tgt_vocab, src_padding, tgt_padding

def optionally_load_pretrained_vectors(model, path):
  model.embeddings.load_pretrained_vectors(path)

def get_dataloader(args, vocab_fields):
  train_data_file = os.path.join(args['data']['root'], args['data']['train_path']) #"squad_cp/data.train.0.pt"
  valid_data_file = os.path.join(args['data']['root'], args['data']['valid_path']) # "squad_cp/data.valid.0.pt"
  train_iter = onmt.inputters.inputter.DatasetLazyIter(dataset_paths=[train_data_file],
                                                       fields=vocab_fields,
                                                       batch_size=50,
                                                       batch_size_multiple=1,
                                                       batch_size_fn=None,
                                                       device=args['device'],
                                                       is_train=True,
                                                       repeat=True,
                                                       pool_factor=1)

  valid_iter = onmt.inputters.inputter.DatasetLazyIter(dataset_paths=[valid_data_file],
                                                       fields=vocab_fields,
                                                       batch_size=10,
                                                       batch_size_multiple=1,
                                                       batch_size_fn=None,
                                                       device=args['device'],
                                                       is_train=False,
                                                       repeat=False,
                                                       pool_factor=1)
  return train_iter, valid_iter

def get_model(args, train, src_padding, tgt_padding):
  encoder = onmt.encoders.RNNEncoder(hidden_size=args['rnn']['rnn_size'], num_layers=1,
                                     rnn_type=args['rnn']['rnn_type'], bidirectional=args['rnn']['bidirectional'],
                                     embeddings=onmt.modules.Embeddings(args['embeddings']['size'], args['vocab_size'], word_padding_idx=src_padding),
                                     use_bridge=True)
  decoder = onmt.decoders.decoder.InputFeedRNNDecoder(
        hidden_size=args['rnn']['rnn_size'], num_layers=1, bidirectional_encoder=args['rnn']['bidirectional'],
        rnn_type=args['rnn']['rnn_type'],
        embeddings=onmt.modules.Embeddings(args['embeddings']['size'], args['vocab_size'], word_padding_idx=tgt_padding),
        copy_attn=True,
        reuse_copy_attn=True,
        copy_attn_type=onmt.modules.GlobalAttention
  )
  if train and args['use_pretrained']:
    logging.info(f"Loading {args['embeddings']['size']}-dimensional pretrained vectors from {args['embeddings']['root']}")
    optionally_load_pretrained_vectors(encoder, os.path.join(args['embeddings']['root'], args['embeddings']['vecs_enc']))
    optionally_load_pretrained_vectors(decoder, os.path.join(args['embeddings']['root'], args['embeddings']['vecs_dec']))

  model = onmt.models.model.NMTModel(encoder, decoder)
  model.generator = onmt.modules.CopyGenerator(args['rnn']['rnn_size'], args['vocab_size'], 1)
  model.to(args['device'])

  return model

def get_loss(args, model, vocab_fields):
  loss = onmt.modules.CopyGeneratorLossCompute(
      onmt.modules.CopyGeneratorLoss(
              args['vocab_size'], False,
              unk_index=vocab_fields["tgt"].base_field.vocab.stoi[vocab_fields["tgt"].base_field.unk_token],
              ignore_index=vocab_fields["tgt"].base_field.vocab.stoi[vocab_fields["tgt"].base_field.pad_token]),
      model.generator,
      vocab_fields["tgt"].base_field.vocab, True, 0.0)
  loss.to(args['device'])
  return loss

def get_optimizer(args, model):
  torch_optimizer = torch.optim.Adagrad(model.parameters(), lr=args['learning_rate'])
  return onmt.utils.optimizers.Optimizer(torch_optimizer, learning_rate=args['learning_rate'], max_grad_norm=2)

def write_params(model, args):
  torch.save(model.state_dict(), args['save_path'])

def train(args):
  vocab_fields, src_text_field, tgt_text_field, src_vocab, tgt_vocab, src_padding, tgt_padding = loadVocab(os.path.join(args['data']['root'], args['data']['vocab_path']))
  args['vocab_size'] = len(src_vocab)
  model = get_model(args, train, src_padding, tgt_padding)
  train_iter, valid_iter = get_dataloader(args, vocab_fields)
  loss = get_loss(args, model, vocab_fields)
  optim = get_optimizer(args, model)

  if args['reporting']['verbose']:
    report_manager = onmt.utils.ReportMgr(report_every=50, start_time=-1, tensorboard_writer=None)
  else:
    report_manager = None
  trainer = onmt.Trainer(model=model,
                         train_loss=loss,
                         valid_loss=loss,
                         optim=optim,
                         report_manager=report_manager)

  logging.info("Beginning training.")
  trainer.train(train_iter=train_iter,
                train_steps=args['reporting']['num_epochs'],
                valid_iter=valid_iter,
                valid_steps=args['reporting']['report_every'])
  logging.info("Finished training.")
  write_params(model, args)


def eval(args):
  vocab_fields, src_text_field, tgt_text_field, src_vocab, tgt_vocab, src_padding, tgt_padding = loadVocab(os.path.join(args['data']['root'], args['data']['vocab_path']))
  args['vocab_size'] = len(src_vocab)
  model = get_model(args, train, src_padding, tgt_padding)
  model.load_state_dict(torch.load(args['save_path']))
  model.eval()
  train_iter, valid_iter = get_dataloader(args, vocab_fields)
  import onmt.translate
  src_reader = onmt.inputters.str2reader["text"]
  tgt_reader = onmt.inputters.str2reader["text"]
  scorer = onmt.translate.GNMTGlobalScorer(alpha=0.7,
                                           beta=0.,
                                           length_penalty="avg",
                                           coverage_penalty="none")
  gpu = 0 if torch.cuda.is_available() else -1
  translator = onmt.translate.Translator(model=model,
                                         fields=vocab_fields,
                                         src_reader=src_reader,
                                         tgt_reader=tgt_reader,
                                         global_scorer=scorer,
                                         gpu=gpu)
  builder = onmt.translate.TranslationBuilder(data=torch.load(valid_data_file),
                                              fields=vocab_fields)

  for batch in valid_iter:
      trans_batch = translator.translate_batch(
          batch=batch, src_vocabs=[src_vocab],
          attn_debug=False)
      translations = builder.from_batch(trans_batch)
      for trans in translations:
          print(trans.log(0))




def __main__():
  argp = ArgumentParser()
  argp.add_argument('experiment_config')
  argp.add_argument('--eval', action='store_true')
  argp.add_argument('--seed', default=0, type=int)
  cli_args = argp.parse_args()
  if cli_args.seed:
    np.random.seed(cli_args.seed)
    torch.manual_seed(cli_args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
  args = yaml.load(open(cli_args.experiment_config))
  device = "cuda" if torch.cuda.is_available() else "cpu"
  args['device'] = device


  if cli_args.eval:
    eval(args)
  else:
    train(args)

if __name__ == "__main__":
  __main__()
