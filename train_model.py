import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from util import load_model
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Dense
from keras.layers.core import Lambda
from keras import backend as K
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import InputLayer
import argparse
import logging
from util import tag_convert
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from keras.layers import BatchNormalization, Dropout
import os
import json

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--emb', default=None)
    parser.add_argument('--outdir', default='./')
    # parser.add_argument('--maxlen', type=int, default=None)
    # parser.add_argument('--num_words', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--pos', choices=['none', 'universal', 'ptb'], default='ptb')
    parser.add_argument('--test_size', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=123)

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    MAX_LEN = None
    NUM_WORDS = 15000
    DEFAULT_EMB_SIZE = 300

    label_encoder = LabelEncoder()
    tokenizer = Tokenizer(NUM_WORDS, filters='\t\n', lower=False)

    logger.info("loading training data ...")
    data = pd.read_csv('./data/train_signal_10_oblig2.tsv.gz', sep='\t',
                       compression='gzip', usecols=['text', 'source'])
    logger.info("done")

    x_data, y_data = data['text'], data['source']

    logger.info("preprocessing data ...")

    y_data = label_encoder.fit_transform(y_data)
    y_data = to_categorical(y_data)

    if args.pos in ('none'):
        x_data = [' '.join([word.split('_')[0] for word in text.split()]) for text in x_data]
    elif args.pos in ('universal'):
        x_data = [' '.join([word.split('_')[0] + '_' + tag_convert(word.split('_')[-1]) for word in text.split()]) for text in x_data]

    tokenizer.fit_on_texts(x_data)
    x_data = tokenizer.texts_to_sequences(x_data)
    x_data = pad_sequences(x_data, maxlen=MAX_LEN)

    # num_words = args.num_words if args.num_words else len(tokenizer.word_index)
    max_len = MAX_LEN if MAX_LEN else x_data.shape[1]
    num_words = NUM_WORDS
    # max_len = MAX_LEN

    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=args.test_size, random_state=args.seed)

    logger.info("done")

    if args.emb:

        logger.info("loading embedding model...")
        emb_model = load_model(args.emb)
        logger.info("done")

        logger.info("building keras embedding layer...")
        embedding_matrix = np.zeros((num_words + 1, emb.wv.vector_size))
        oov_count = 0

        for i in range(1, num_words + 1):
            word = tokenizer.index_word[i]
            if word in emb.wv.vocab:
                embedding_matrix[i] = emb.wv.get_vector(word)
            else:
                oov_count += 1

        if oov_count:
            logger.warn("%d words were not in model vocabulary", oov_count)

        emb_layer = Embedding(num_words + 1, emb.wv.vector_size, input_length=max_len,
                              weights=[embedding_matrix], trainable=False)

        logger.info("done")

    else:
        logger.info("no embedding file, initialized randomly and trainable")
        emb_layer = Embedding(num_words + 1, DEFAULT_EMB_SIZE, trainable=True, input_length=max_len)

    model = Sequential()
    model.add(emb_layer)
    model.add(Lambda(lambda x: K.mean(x, axis=1)))
    # model.add(Dense(256, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    tmp_model_path = os.path.join(args.outdir, 'tmp.model')

    callbacks = [
        ModelCheckpoint(tmp_model_path),
        TensorBoard(log_dir=os.path.join(args.outdir, 'logs')),
        EarlyStopping(patience=3)
    ]

    model.fit(x_train, y_train, epochs=args.epochs, validation_data=(x_test, y_test), callbacks=callbacks)

    model.load_weights(tmp_model_path)

    logger.info("Finished training model")
    logger.info("Evaluating on developement set...")
    y_true = np.argmax(data_y, axis=1)
    y_pred = np.argmax(model.predict(x_test), axis=1)
    logger.info("done")

    labels = label_encoder.inverse_transform(np.unique(y_test))
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred)
    avg_prec, avg_rec, avg_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    C = confusion_matrix(y_true, y_pred)

    metrics = {
        "labels": labels.tolist(),
        "accuracy": acc,
        "precision": prec.tolist(),
        "recall": rec.tolist(),
        "f1": f1.tolist(),
        "avg_precision": avg_prec,
        "avg_recall": avg_rec,
        "avg_f1": avg_f1,
        "confusion_matrix": C.tolist()
    }

    with open(os.path.join(args.outdir, 'metrics.json')) as f:
        json.dump(metrics, f)
