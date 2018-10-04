import pickle
import argparse
import logging
import json
import sys
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from util import get_metrics

if __name__ == '__main__':

    MAX_LEN = 1000

    parser = argparse.ArgumentParser(description="""
    Script that loads model and it's tokenizer,
    and evaluates it on training data.
    Computes precision recall and f1 both per class and macro average.
    Saves the computed metrics to the file or prints to the stdout.
    """)
    parser.add_argument('model', help='classifier model')
    parser.add_argument('tokenizer', help='tokenizer used with model')
    parser.add_argument('test_data', help='data to test on')
    parser.add_argument('--output_file', help='file where to save the results')

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load the model,
    # and the corresponding tokenizer.
    logging.info("loading model...")
    with open(args.tokenizer, 'rb') as f:
        tokenizer = pickle.load(f)

    model = load_model(args.model)
    logging.info("done")
    print(model.summary())

    # Load and preprocess dataset
    logging.info("loading and preprocessing dataset...")
    test_dataset = pd.read_csv(args.test_data, sep='\t', compression="gzip")
    label_encoder = LabelEncoder()

    (x_test, y_test) = test_dataset['text'], test_dataset['source']

    y_test = label_encoder.fit_transform(y_test)
    y_test = to_categorical(y_test)

    x_test = tokenizer.texts_to_sequences(x_test)
    x_test = pad_sequences(x_test, maxlen=MAX_LEN)
    logging.info("done")

    # Evaluate model on test set
    logging.info("evaluating model...")
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(model.predict(x_test), axis=1)

    metrics = get_metrics(y_pred, y_true)
    metrics['labels'] = label_encoder.inverse_transform(np.unique(y_true)).tolist()
    logging.info("done")

    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(metrics, f)
    else:
        print(metrics)
