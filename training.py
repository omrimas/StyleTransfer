import model
import voc
import torch
from torch import nn
import torch.optim as optim
import random
from helpers import *
import time
import os
import pickle

# model params
INPUT_SIZE = 300
HIDDEN_SIZE = 512
NUM_LAYERS = 2
LEARNING_RATE = 0.0001

MAX_LENGTH = 100

TRAINING_DATA_PATH = os.path.join("data", "training")
TRAINING_DATA_FILES = ["disgust", "joy"]
VOC_PICKLE = os.path.join("pickles", "voc.pkl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loadVoc():
    f_voc = open(VOC_PICKLE, "rb")
    vocab = pickle.load(f_voc)
    return vocab


def getLines():
    lines = {}
    for train_file in TRAINING_DATA_FILES:
        file_path = os.path.join(TRAINING_DATA_PATH, train_file)
        lines[train_file] = open(file_path, encoding='utf-8'). \
            read().strip().split('\n')
    return lines


def indexesFromSentence(vocab, sentence):
    return [vocab.word2index[w] for w in sentence.split()]


def tensorFromSentence(vocab, sentence):
    indexes = indexesFromSentence(vocab, sentence)
    indexes.append(voc.EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device)


lines = getLines()
vocab = loadVoc()

teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          max_length=MAX_LENGTH):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    loss = 0

    # encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    # for ei in range(input_length):
    #     encoder_output, encoder_hidden = encoder(
    #         input_tensor[ei], encoder_hidden)
    #     encoder_outputs[ei] = encoder_output[0, 0]

    encoder_hidden = encoder.initHidden()
    encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)

    decoder_input = torch.tensor([[voc.SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    # use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    use_teacher_forcing = False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == voc.EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def trainIters(encoder, decoders, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizers = {}
    for train_file, decoder in decoders.items():
        decoder_optimizers[train_file] = optim.SGD(decoder.parameters(), lr=learning_rate)

    training_sentences = {}
    for train_file in TRAINING_DATA_FILES:
        training_sentences[train_file] = [tensorFromSentence(vocab, random.choice(lines[train_file]))
                              for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        for train_file in TRAINING_DATA_FILES:
            training_sentence = training_sentences[train_file][iter - 1]
            input_tensor = training_sentence
            target_tensor = training_sentence.view(-1, 1)

            loss = train(input_tensor, target_tensor, encoder,
                         decoders[train_file], encoder_optimizer, decoder_optimizers[train_file], criterion)
            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                             iter, iter / n_iters * 100, print_loss_avg))

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

    showPlot(plot_losses)


hidden_size = 256
encoder1 = model.EncoderRNN(vocab.n_words, hidden_size, device).to(device)
decoders = {}
for train_file in TRAINING_DATA_FILES:
    decoders[train_file] = model.Decoder1RNN(hidden_size, vocab.n_words, device).to(device)

trainIters(encoder1, decoders, 500, print_every=5000)
