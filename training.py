import model
import voc
import torch
from torch import nn
import torch.optim as optim

# model params
INPUT_SIZE = 300
HIDDEN_SIZE = 512
NUM_LAYERS = 2
LEARNING_RATE = 0.0001

IS_CUDA = False


def index_sentence(vocab, sentence):
    return [vocab.word2index[w] for w in sentence.split()]


# build vocabulary
vocab = voc.Voc("depression")
vocab.add_sentence("i love eating bananas")

model = model.LSTMAE(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, IS_CUDA)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

input_sentence = "i love eating"
indexed_sentence = index_sentence(vocab, input_sentence)
model_input = torch.tensor([indexed_sentence])
model_target = torch.tensor(indexed_sentence)
model.zero_grad()
model_output = model(model_input)
loss = criterion(model_output.view(-1, vocab.num_words), model_target)
print(loss.item())
loss.backward()
optimizer.step()
