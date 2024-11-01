#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This code is based on the tutorial by Sean Robertson <https://github.com/spro/practical-pytorch> found here:
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

Students *MAY NOT* view the above tutorial or use it as a reference in any way. 
"""


from __future__ import unicode_literals, print_function, division

import argparse
import logging
import random
import time
from io import open

import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.translate.bleu_score import corpus_bleu
from torch import optim


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = "<SOS>"
EOS_token = "<EOS>"
PAD_token = "<PAD>"

SOS_index = 0
EOS_index = 1
PAD_index = 2 
MAX_LENGTH = 15


class Vocab:
    """ This class handles the mapping between the words and their indices
    """
    def __init__(self, lang_code):
        self.lang_code = lang_code
        self.word2index = {SOS_token: SOS_index, EOS_token: EOS_index, PAD_token: PAD_index}
        self.word2count = {}
        self.index2word = {SOS_index: SOS_token, EOS_index: EOS_token, PAD_index: PAD_token}
        self.n_words = 3 

    def add_sentence(self, sentence):
        for word in sentence.strip().split(' '):
            self._add_word(word)

    def _add_word(self, word):
        word = word.strip()
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


######################################################################


def split_lines(input_file):
    """split a file like:
    first src sentence|||first tgt sentence
    second src sentence|||second tgt sentence
    into a list of things like
    [("first src sentence", "first tgt sentence"), 
     ("second src sentence", "second tgt sentence")]
    """
    logging.info("Reading lines of %s...", input_file)
    # Read the file and split into lines
    lines = open(input_file, encoding='utf-8').read().strip().split('\n')
    # Split every line into pairs
    pairs = [l.strip().split('|||') for l in lines]
    # Ensure each pair has exactly two elements
    pairs = [pair for pair in pairs if len(pair) == 2]
    return pairs


def make_vocabs(src_lang_code, tgt_lang_code, train_file):
    """ Creates the vocabs for each of the languages based on the training corpus.
    """
    src_vocab = Vocab(src_lang_code)
    tgt_vocab = Vocab(tgt_lang_code)

    train_pairs = split_lines(train_file)

    for pair in train_pairs:
        src_vocab.add_sentence(pair[0])
        tgt_vocab.add_sentence(pair[1])

    logging.info('%s (src) vocab size: %s', src_vocab.lang_code, src_vocab.n_words)
    logging.info('%s (tgt) vocab size: %s', tgt_vocab.lang_code, tgt_vocab.n_words)

    return src_vocab, tgt_vocab

######################################################################

def tensor_from_sentence(vocab, sentence):
    """creates a tensor from a raw sentence
    """
    indexes = []
    for word in sentence.strip().split():
        try:
            indexes.append(vocab.word2index[word])
        except KeyError:
            indexes.append(vocab.word2index.get("<UNK>", PAD_index)) 
    indexes.append(EOS_index)
    return torch.tensor(indexes, dtype=torch.long, device=device)

def pad_sequences(sequences, max_length=None, padding_value=PAD_index):
    """Pad a list of sequences to the same length."""
    if not max_length:
        max_length = max([len(seq) for seq in sequences])
    padded_sequences = torch.full((len(sequences), max_length), padding_value, dtype=torch.long, device=device)
    for i, seq in enumerate(sequences):
        length = len(seq)
        padded_sequences[i, :length] = seq
    return padded_sequences

def tensors_from_pair(src_vocab, tgt_vocab, pair):
    """creates tensors from a raw sentence pair
    """
    input_tensor = tensor_from_sentence(src_vocab, pair[0])
    target_tensor = tensor_from_sentence(tgt_vocab, pair[1])
    return input_tensor, target_tensor

######################################################################

class EncoderRNN(nn.Module):
    """Encoder RNN using PyTorch's nn.LSTM."""
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        # Use a bidirectional LSTM
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, input_seqs, input_lengths):
        batch_size = input_seqs.size(0)
        embedded = self.embedding(input_seqs)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_outputs, (h_n, c_n) = self.lstm(packed_embedded)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True) 
        return outputs, (h_n, c_n)

######################################################################

class AttnDecoderRNN(nn.Module):
    """Decoder RNN using PyTorch's nn.LSTMCell with attention."""
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size 
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
       
        self.lstm_cell = nn.LSTMCell(hidden_size + hidden_size * 2, hidden_size)

        # Attention layers
        self.attn_Wa = nn.Linear(hidden_size, hidden_size)
        self.attn_Ua = nn.Linear(hidden_size * 2, hidden_size)
        self.attn_v = nn.Linear(hidden_size, 1, bias=False)

        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, encoder_outputs, mask):

        batch_size = input.size(0)
        embedded = self.embedding(input) 
        embedded = self.dropout(embedded)

        s_prev = hidden[0]  
        s_prev_expanded = s_prev.unsqueeze(1) 

        attn_scores = self.attn_v(torch.tanh(
            self.attn_Wa(s_prev_expanded) + self.attn_Ua(encoder_outputs)
        )).squeeze(2)

        # Apply mask to attention scores
        attn_scores.data.masked_fill_(mask == 0, -1e9)

        attn_weights = F.softmax(attn_scores, dim=1)  

        c_i = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  

        lstm_input = torch.cat((embedded, c_i), dim=1)  

        h_i, c_i = self.lstm_cell(lstm_input, hidden)  

        output = self.out(h_i)  
        log_softmax = F.log_softmax(output, dim=1)

        return log_softmax, (h_i, c_i), attn_weights

    def get_initial_hidden_state(self, batch_size):
        return (torch.zeros(batch_size, self.hidden_size, device=device),
                torch.zeros(batch_size, self.hidden_size, device=device))

######################################################################

def create_masks(input_lengths, max_length):
    """Creates masks for sequences based on their lengths."""
    batch_size = len(input_lengths)
    mask = torch.zeros(batch_size, max_length, dtype=torch.bool, device=device)
    for i, length in enumerate(input_lengths):
        mask[i, :length] = True
    return mask 

def train_batch(input_tensor, target_tensor, input_lengths, target_lengths, encoder, decoder, optimizer, criterion, max_length=MAX_LENGTH):
    batch_size = input_tensor.size(0)
    encoder_hidden = None 

    encoder.train()
    decoder.train()

    optimizer.zero_grad()

    # Sort input sequences by lengths in descending order
    input_lengths, perm_idx = input_lengths.sort(0, descending=True)
    input_tensor = input_tensor[perm_idx]
    target_tensor = target_tensor[perm_idx]
    target_lengths = target_lengths[perm_idx]

    encoder_outputs, encoder_hidden = encoder(input_tensor, input_lengths)

    # Create mask for attention
    encoder_mask = create_masks(input_lengths, encoder_outputs.size(1)) 

    decoder_input = torch.tensor([SOS_index] * batch_size, device=device)  
    decoder_hidden = decoder.get_initial_hidden_state(batch_size)

    max_target_length = max(target_lengths)
    loss = 0

    # Create mask for target sequences
    target_mask = create_masks(target_lengths, max_target_length) 

    for di in range(max_target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs, encoder_mask)
        target = target_tensor[:, di]

        non_pad_mask = target != PAD_index
        if non_pad_mask.sum().item() == 0:
            continue
        loss += criterion(decoder_output[non_pad_mask], target[non_pad_mask])
        decoder_input = target  

    loss.backward()
    optimizer.step()

    return loss.item() / max_target_length

######################################################################

def translate(encoder, decoder, sentence, src_vocab, tgt_vocab, max_length=MAX_LENGTH, beam_width=5):
    """
    Runs translation using beam search, returns the output and attention.
    """
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        input_tensor = tensor_from_sentence(src_vocab, sentence).unsqueeze(0)  
        input_length = torch.tensor([input_tensor.size(1)], device=device)
        encoder_outputs, encoder_hidden = encoder(input_tensor, input_length)

        encoder_mask = create_masks(input_length, encoder_outputs.size(1))  

        decoder_hidden = decoder.get_initial_hidden_state(1)

        beam = [([SOS_index], decoder_hidden, 0.0)]

        completed_sequences = []

        for _ in range(max_length):
            new_beam = []
            for seq, hidden, score in beam:
                decoder_input = torch.tensor([seq[-1]], device=device) 

                if seq[-1] == EOS_index:
                    
                    completed_sequences.append((seq, score))
                    continue

                decoder_output, hidden, decoder_attention = decoder(
                    decoder_input, hidden, encoder_outputs, encoder_mask)

           
                topv, topi = decoder_output.data.topk(beam_width)
                for i in range(beam_width):
                    next_word = topi[0][i].item()
                    new_seq = seq + [next_word]
                    new_score = score + topv[0][i].item()
                    new_beam.append((new_seq, hidden, new_score))

        
            new_beam = sorted(new_beam, key=lambda x: x[2], reverse=True)[:beam_width]
            beam = new_beam

            if all(seq[-1] == EOS_index for seq, _, _ in beam):
                completed_sequences.extend([(seq, score) for seq, _, score in beam if seq[-1] == EOS_index])
                break

        if not completed_sequences:
            completed_sequences = beam

        best_sequence, best_score = max(completed_sequences, key=lambda x: x[1])

        decoded_words = [tgt_vocab.index2word.get(idx, "<UNK>") for idx in best_sequence[1:]] 

        attentions = None  

        return decoded_words, attentions


################################################################
def translate_sentences(encoder, decoder, pairs, src_vocab, tgt_vocab, max_num_sentences=None, max_length=MAX_LENGTH, beam_width=5):
    output_sentences = []
    for pair in pairs[:max_num_sentences]:
        output_words, _ = translate(encoder, decoder, pair[0], src_vocab, tgt_vocab, max_length=max_length, beam_width=beam_width)
        output_sentence = ' '.join(output_words)
        output_sentences.append(output_sentence)
    return output_sentences


######################################################################

def translate_random_sentence(encoder, decoder, pairs, src_vocab, tgt_vocab, n=1, beam_width=5):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, _ = translate(encoder, decoder, pair[0], src_vocab, tgt_vocab, beam_width=beam_width)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


######################################################################

def show_attention(input_sentence, output_words, attentions, save_path='attention.png'):
    """
    Displays the attention weights as a heatmap.

    Args:
        input_sentence (str): The input sentence (source language).
        output_words (list): The list of output words (target language).
        attentions (torch.Tensor): The attention weights matrix of shape (output_length, input_length).
        save_path (str): The path to save the attention plot.
    """


    attentions = attentions.cpu().numpy()

    assert attentions.shape[0] == len(output_words), \
        f"Attention matrix rows ({attentions.shape[0]}) != number of output words ({len(output_words)})"
    assert attentions.shape[1] == len(input_sentence.strip().split()) + 1, \
        f"Attention matrix columns ({attentions.shape[1]}) != number of input words + 1 ({len(input_sentence.strip().split()) + 1})"

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    cax = ax.matshow(attentions, cmap='bone')
    fig.colorbar(cax)

    input_words = input_sentence.strip().split(' ') + [EOS_token]
    output_words = output_words  

    ax.set_xticks(range(len(input_words)))
    ax.set_yticks(range(len(output_words)))

    ax.set_xticklabels(input_words, rotation=90, fontsize=12)
    ax.set_yticklabels(output_words, fontsize=12)

    ax.set_xlabel('Input Sentence', fontsize=14)
    ax.set_ylabel('Output Words', fontsize=14)

    plt.tight_layout()

    plt.savefig(save_path)
    plt.close(fig) 

    logging.info(f'Attention plot saved to {save_path}')

def translate_and_show_attention(input_sentence, encoder1, decoder1, src_vocab, tgt_vocab):
    output_words, attentions = translate(
        encoder1, decoder1, input_sentence, src_vocab, tgt_vocab)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    show_attention(input_sentence, output_words, attentions)

def clean(strx):
    """
    input: string with bpe, EOS
    output: cleaned string without bpe, EOS
    """
    return ' '.join(strx.replace('@@ ', '').replace(EOS_token, '').strip().split())

######################################################################

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hidden_size', default=256, type=int,
                    help='hidden size of encoder/decoder, also word vector size')
    ap.add_argument('--n_iters', default=5000, type=int,
                    help='total number of examples to train on')
    ap.add_argument('--print_every', default=5000, type=int,
                    help='print loss info every this many training examples')
    ap.add_argument('--checkpoint_every', default=10000, type=int,
                    help='write out checkpoint every this many training examples')
    ap.add_argument('--initial_learning_rate', default=0.001, type=float,
                    help='initial learning rate')
    ap.add_argument('--src_lang', default='fr',
                    help='Source (input) language code, e.g. "fr"')
    ap.add_argument('--tgt_lang', default='en',
                    help='Target (output) language code, e.g. "en"')
    ap.add_argument('--train_file', default='data/fren.train.bpe',
                    help='training file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence')
    ap.add_argument('--dev_file', default='data/fren.dev.bpe',
                    help='dev file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence')
    ap.add_argument('--test_file', default='data/fren.test.bpe',
                    help='test file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence' +
                         ' (for test, target is ignored)')
    ap.add_argument('--out_file', default='translations',
                    help='output file for test translations')
    ap.add_argument('--load_checkpoint', nargs=1,
                    help='checkpoint file to start from')
    ap.add_argument('--batch_size', default=64, type=int,
                    help='batch size for training')
    args = ap.parse_args()


    if args.load_checkpoint is not None:
        state = torch.load(args.load_checkpoint[0])
        iter_num = state['iter_num']
        src_vocab = state['src_vocab']
        tgt_vocab = state['tgt_vocab']
    else:
        iter_num = 0
        src_vocab, tgt_vocab = make_vocabs(args.src_lang,
                                           args.tgt_lang,
                                           args.train_file)

    encoder = EncoderRNN(src_vocab.n_words, args.hidden_size).to(device)
    decoder = AttnDecoderRNN(args.hidden_size, tgt_vocab.n_words, dropout_p=0.1).to(device)

    if args.load_checkpoint is not None:
        encoder.load_state_dict(state['enc_state'])
        decoder.load_state_dict(state['dec_state'])


    train_pairs = split_lines(args.train_file)
    dev_pairs = split_lines(args.dev_file)
    test_pairs = split_lines(args.test_file)


    params = list(encoder.parameters()) + list(decoder.parameters()) 
    optimizer = optim.Adam(params, lr=args.initial_learning_rate)
    criterion = nn.NLLLoss(ignore_index=PAD_index)

    if args.load_checkpoint is not None:
        optimizer.load_state_dict(state['opt_state'])

    start = time.time()
    print_loss_total = 0  

    batch_size = args.batch_size
    n_iters = args.n_iters
    iter_num = 0

    while iter_num < n_iters:
        random.shuffle(train_pairs)
        for batch_start in range(0, len(train_pairs), batch_size):
            iter_num += 1
            batch_pairs = train_pairs[batch_start:batch_start + batch_size]
            batch_size_actual = len(batch_pairs)
            input_tensors = [tensor_from_sentence(src_vocab, pair[0]) for pair in batch_pairs]
            target_tensors = [tensor_from_sentence(tgt_vocab, pair[1]) for pair in batch_pairs]
            input_lengths = torch.tensor([len(tensor) for tensor in input_tensors], dtype=torch.long, device=device)
            target_lengths = torch.tensor([len(tensor) for tensor in target_tensors], dtype=torch.long, device=device)

            max_input_length = input_lengths.max().item()
            max_target_length = target_lengths.max().item()

            input_padded = pad_sequences(input_tensors, max_input_length)  
            target_padded = pad_sequences(target_tensors, max_target_length) 

            loss = train_batch(input_padded, target_padded, input_lengths, target_lengths, encoder,
                             decoder, optimizer, criterion)
            print_loss_total += loss

            if iter_num % args.checkpoint_every == 0:
                state = {'iter_num': iter_num,
                         'enc_state': encoder.state_dict(),
                         'dec_state': decoder.state_dict(),
                         'opt_state': optimizer.state_dict(),
                         'src_vocab': src_vocab,
                         'tgt_vocab': tgt_vocab,
                         }
                filename = 'state_%010d.pt' % iter_num
                torch.save(state, filename)
                logging.debug('wrote checkpoint to %s', filename)

            if iter_num % args.print_every == 0:
                print_loss_avg = print_loss_total / args.print_every
                print_loss_total = 0
                logging.info('time since start:%s (iter:%d iter/n_iters:%d%%) loss_avg:%.4f',
                             time.time() - start,
                             iter_num,
                             iter_num / args.n_iters * 100,
                             print_loss_avg)

                translate_random_sentence(encoder, decoder, dev_pairs, src_vocab, tgt_vocab, n=2, beam_width=5)
                translated_sentences = translate_sentences(encoder, decoder, dev_pairs, src_vocab, tgt_vocab, max_num_sentences=100, beam_width=5)

                references = [[clean(pair[1]).split(), ] for pair in dev_pairs[:len(translated_sentences)]]
                candidates = [clean(sent).split() for sent in translated_sentences]
                dev_bleu = corpus_bleu(references, candidates)
                logging.info('Dev BLEU score: %.2f', dev_bleu)

            if iter_num >= n_iters:
                break

    translated_sentences = translate_sentences(encoder, decoder, test_pairs, src_vocab, tgt_vocab)
    with open(args.out_file, 'wt', encoding='utf-8') as outf:
        for sent in translated_sentences:
            outf.write(clean(sent) + '\n')

    translate_and_show_attention("on p@@ eu@@ t me faire confiance .", encoder, decoder, src_vocab, tgt_vocab)
    translate_and_show_attention("j en suis contente .", encoder, decoder, src_vocab, tgt_vocab)
    translate_and_show_attention("vous etes tres genti@@ ls .", encoder, decoder, src_vocab, tgt_vocab)
    translate_and_show_attention("c est mon hero@@ s ", encoder, decoder, src_vocab, tgt_vocab)


if __name__ == '__main__':
    main()
