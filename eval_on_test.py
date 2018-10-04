import sys
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('tokenizer')
    parser.add_argument('test_data')

    args = parser.parse_args()

    with open(args.tokenizer, 'rb') as f:
        tokenizer = pickle.load(f)

    model = load_model(args.model)
    print(model.summary())

    label_encoder = LabelEncoder()
    print('Loading the test set...')
    test_dataset = pd.read_csv(args.test_data, sep='\t', compression="gzip")
    print('Finished loading the test set')

    (x_test, y_test) = test_dataset['text'], test_dataset['source']

    print(len(x_test), 'test texts')

    # print('Average test text length: {0:.{1}f} words'.format(np.mean(list(map(len, x_test.str.split()))), 1))

    y_test = label_encoder.fit_transform(y_test)
    y_test = to_categorical(y_test)

    # classes = sorted(list(set(y_test)))
    # num_classes = len(classes)
    # print(num_classes, 'classes')

    # print('===========================')
    # print('Class distribution in the testing data:')
    # print(test_dataset.groupby('source').count())
    # print('===========================')

    # We can remove PoS tags if we think we don't need them:
    # x_test = [' '.join([word.split('_')[0] for word in text.split()]) for text in x_test]

    x_test = tokenizer.texts_to_sequences(x_test)
    x_test = pad_sequences(x_test, maxlen=1000)
    # print('Vectorizing text data...')
    # tokenized_test = tokenizer.texts_to_matrix(x_test, mode='binary')  # Count or binary BOW?
    # print('Test data shape:', tokenized_test.shape)

    # Converting text labels to indexes
    # y_test = [classes.index(i) for i in y_test]

    # Convert indexes to binary class matrix (for use with categorical_crossentropy loss)
    # y_test = to_categorical(y_test, num_classes)
    # print('Test labels shape:', y_test.shape)

    print('===========================')
    print('Evaluation:')
    print('===========================')

    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(model.predict(x_test), axis=1)

    labels = label_encoder.inverse_transform(np.unique(y_true))
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

    print(metrics)