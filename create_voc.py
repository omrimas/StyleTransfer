import voc
import os
import time
import pickle
import logging

logging.basicConfig(format='%(asctime)s %(message)s')
logging.getLogger().setLevel(logging.INFO)

TRAINING_DATA_PATH = os.path.join("data", "training")
TRAINING_DATA_FILES = ["disgust", "joy"]

VOC_PICKLE = os.path.join("pickles", "voc.pkl")

start = time.time()
vocab = voc.Voc("StyleTransfer")
logging.info('Building vocabulary...')
for train_file in TRAINING_DATA_FILES:
    file_path = os.path.join(TRAINING_DATA_PATH, train_file)
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            vocab.addSentence(line.strip())

logging.info('Saving vocabulary to pickle...')
f = open(VOC_PICKLE, "wb")
pickle.dump(vocab, f)
f.close()
logging.info('Done!')
