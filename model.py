import numpy as np
import torch
from torch import nn
from torch.autograd import Variable


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, isCuda):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.isCuda = isCuda
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # self.init_weights()
        # self.relu = nn.ReLU()

    def init_weights(self):
        nn.init.xavier_uniform_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        nn.init.xavier_uniform_(self.lstm.weight_hh_l0, gain=np.sqrt(2))

        initrange = 0.1
        self.word_embeddings.weight.data.uniform_(-initrange, initrange)

    def forward(self, input):
        hidden = self.init_hidden(input.size(0))
        encoded_input, hidden = self.lstm(input, hidden)
        return encoded_input

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.num_layers, bsz, self.hidden_size).zero_()),
                Variable(weight.new(self.num_layers, bsz, self.hidden_size).zero_()))


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, isCuda):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.isCuda = isCuda
        self.lstm = nn.LSTM(hidden_size, output_size, num_layers, batch_first=True)

        self.hidden2word = nn.Linear(output_size, 5)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        nn.init.xavier_uniform_(self.lstm.weight_hh_l0, gain=np.sqrt(2))

        initrange = 0.1
        self.hidden2word.bias.data.fill_(0)
        self.hidden2word.weight.data.uniform_(-initrange, initrange)

    def forward(self, encoded_input):
        hidden = self.init_hidden(encoded_input.size(0))
        decoded_output, hidden = self.lstm(encoded_input, hidden)
        batch_size, time_size, features_size = decoded_output.size()
        voc_space_output = self.hidden2word(decoded_output.view(batch_size * time_size, features_size))
        return voc_space_output.view(batch_size, time_size, 5)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.num_layers, bsz, self.output_size).zero_()),
                Variable(weight.new(self.num_layers, bsz, self.output_size).zero_()))


class LSTMAE(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, isCuda):
        super(LSTMAE, self).__init__()
        self.embed = nn.Embedding(5, input_size)
        self.encoder = EncoderRNN(input_size, hidden_size, num_layers, isCuda)
        self.decoder = DecoderRNN(hidden_size, input_size, num_layers, isCuda)

    def forward(self, input):
        embedded_input = self.embed(input)
        encoded_input = self.encoder(embedded_input)
        decoded_output = self.decoder(encoded_input)
        return decoded_output
