import gensim
import zipfile
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

conversion_table = {
    "#": "SYM",
    "$": "SYM",
    "''": "PUNCT",
    ",": "PUNCT",
    "-LRB-": "PUNCT",
    "-RRB-": "PUNCT",
    ".": "PUNCT",
    ":": "PUNCT",
    "AFX": "ADJ",
    "CC": "CCONJ",
    "CD": "NUM",
    "DT": "DET",
    "EX": "PRON",
    "FW": "X",
    "HYPH": "PUNCT",
    "IN": "ADP",
    "JJ": "ADJ",
    "JJR": "ADJ",
    "JJS": "ADJ",
    "LS": "X",
    "MD": "VERB",
    "NIL": "X",
    "NN": "NOUN",
    "NNP": "PROPN",
    "NNPS": "PROPN",
    "NNS": "NOUN",
    "PDT": "DET",
    "POS": "PART",
    "PRP": "PRON",
    "PRP$": "DET",
    "RB": "ADV",
    "RBR": "ADV",
    "RBS": "ADV",
    "RP": "ADP",
    "SYM": "SYM",
    "TO": "PART",
    "UH": "INTJ",
    "VB": "VERB",
    "VBD": "VERB",
    "VBG": "VERB",
    "VBN": "VERB",
    "VBP": "VERB",
    "VBZ": "VERB",
    "WDT": "DET",
    "WP": "PRON",
    "WP$": "DET",
    "WRB": "ADV",
    "``": "PUNCT"
}


def load_model(embeddings_file):
    """Loads word embeddings model from file."""
    if embeddings_file.endswith('.bin.gz') or embeddings_file.endswith('.bin'):  # Binary word2vec format
        return gensim.models.KeyedVectors.load_word2vec_format(
            embeddings_file, binary=True, unicode_errors='replace')

    elif embeddings_file.endswith('.txt.gz') or embeddings_file.endswith('.txt') or\
            embeddings_file.endswith('.vec.gz') or embeddings_file.endswith('.vec'):  # Text word2vec format
        return gensim.models.KeyedVectors.load_word2vec_format(
            embeddings_file, binary=False, unicode_errors='replace')

    elif embeddings_file.endswith('.zip'):  # ZIP archive from the NLPL vector repository
        with zipfile.ZipFile(embeddings_file, "r") as archive:
            stream = archive.open("model.txt")
            return gensim.models.KeyedVectors.load_word2vec_format(stream, binary=False, unicode_errors='replace')

    else:  # Native Gensim format?
        return gensim.models.Word2Vec.load(embeddings_file)


def tag_convert(token):
    """Converts tag from PTB tagset to Universal tagset"""
    return conversion_table.get(token, '')
