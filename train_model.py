import os
import json
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from util import load_model, get_metrics
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
from sklearn.utils.class_weight import compute_sample_weight

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="""
        Trains the classifier.
    """)

    parser.add_argument('--emb', default=None, help="embeddings model to use, if not provided embeddings are trained")
    parser.add_argument('--outdir', default='./', help="where to output metrics, logs and save the model")
    parser.add_argument('--save', action='store_true', default=False, help="wether to save the model")
    parser.add_argument('--epochs', type=int, default=10, help="number of epochs")
    parser.add_argument('--pos', choices=['none', 'universal', 'ptb'], default='ptb', help="which tags to use")
    parser.add_argument('--test_size', type=float, default=0.1, help="size of validation set")
    parser.add_argument('--seed', type=int, default=123, help="seed for random shuffling")

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    MAX_LEN = 1000
    NUM_WORDS = 3000
    DEFAULT_EMB_SIZE = 100

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
    # x_data = tokenizer.sequences_to_matrix(x_data, mode='binary')

    # num_words = args.num_words if args.num_words else len(tokenizer.word_index)
    max_len = MAX_LEN if MAX_LEN else x_data.shape[1]
    num_words = NUM_WORDS
    # max_len = MAX_LEN
    logger.info("sequences padded to: %s" % max_len)

    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=args.test_size, random_state=args.seed)

    logger.info("done")

    if args.emb:

        logger.info("loading embedding model...")
        emb_model = load_model(args.emb)
        logger.info("done")

        logger.info("building keras embedding layer...")
        embedding_matrix = np.zeros((num_words + 1, emb_model.wv.vector_size))
        oov_count = 0

        for i in range(1, num_words + 1):
            word = tokenizer.index_word[i]
            if word in emb_model.wv.vocab:
                embedding_matrix[i] = emb_model.wv.get_vector(word)
            else:
                oov_count += 1

        if oov_count:
            logger.warn("%d words were not in model vocabulary", oov_count)

        emb_layer = Embedding(num_words + 1, emb_model.wv.vector_size, input_length=max_len,
                              weights=[embedding_matrix], trainable=False)
        # emb_layer = Embedding(num_words + 1, emb_model.wv.vector_size, input_length=max_len,
        #                 weights=[embedding_matrix], trainable=True)

        logger.info("done")

    else:
        logger.info("no embedding file, initialized randomly and trainable")
        emb_layer = Embedding(num_words + 1, DEFAULT_EMB_SIZE, trainable=True, input_length=max_len)

    model = Sequential()
    model.add(emb_layer)
    model.add(Lambda(lambda x: K.mean(x, axis=1)))
    model.add(Dense(256, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
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

    # sample_weight = compute_sample_weight('balanced', y_train)

    model.fit(x_train, y_train, epochs=args.epochs, validation_data=(x_test, y_test),
              callbacks=callbacks)

    model.load_weights(tmp_model_path)
    if os.path.exists(tmp_model_path):
        os.remove(tmp_model_path)

    if args.save:
        with open(os.path.join(args.outdir, 'tokenizer.pkl'), "wb") as f:
            pickle.dump(tokenizer, f)

        model.save(os.path.join(args.outdir, 'model.h5'))

    logger.info("Finished training model")

    logger.info("Evaluating on developement set...")
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(model.predict(x_test), axis=1)
    logger.info("done")

    metrics = get_metrics(y_pred, y_true)
    metrics['labels'] = label_encoder.inverse_transform(np.unique(y_true)).tolist()

    with open(os.path.join(args.outdir, 'metrics.json'), "w") as f:
        json.dump(metrics, f)
