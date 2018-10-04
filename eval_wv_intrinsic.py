import os
import sys
import json
import argparse
import gensim
import logging
from util import load_model

analogies_data = './data/analogies/analogies_semantic.txt'
analogies_POS_data = './data/analogies/analogies_semantic_POS.txt'

simlex_data = {
    'total': './data/simlex/simlex.tsv',
    'adj': './data/simlex/simlex_adj.tsv',
    'noun': './data/simlex/simlex_noun.tsv',
    'verb': './data/simlex/simlex_verb.tsv'
}

simlex_data_pos = {
    'total': './data/simlex/simlex_POS.tsv',
    'adj': './data/simlex/simlex_adj_POS.tsv',
    'noun': './data/simlex/simlex_noun_POS.tsv',
    'verb': './data/simlex/simlex_verb_POS.tsv'
}


def eval_on_simlex(model, logger=None, pos=False):
    """Evaluates word vectors on simlex dataset."""
    if logger:
        logger.info('Evaluating on SimLex999...')

    results = {}
    data = simlex_data_pos if pos else simlex_data

    for key, value in data.items():
        pearson, spearman, oov_ratio = model.wv.evaluate_word_pairs(value)
        results[key] = {
            'pearson': pearson,
            'spearman': spearman,
            'oov_ratio': oov_ratio
        }

    if logger:
        logger.info('Finished evaluating on SimLex999')

    return results


def eval_on_analogies(model, logger=None, pos=False):
    "Evaluates word vectors on google analogies dataset."
    if logger:
        logger.info('Evaluating on analogies...')

    data = analogies_POS_data if pos else analogies_data
    total, sections = model.wv.evaluate_word_analogies(data)

    section_names = []
    scores = {}
    for section in sections:
        section_names.append(section['section'])
        correct = len(section['correct'])
        incorrect = len(section['incorrect'])
        if correct + incorrect:
            scores[section['section']] = correct / float(correct + incorrect)

    results = {}
    results['sections'] = section_names
    results['scores'] = scores

    if logger:
        logger.info('Finished evaluating on analogies...')

    return results

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="""
    Evaluates embeddings model on intrinsic evaluation datasets:
    - SimLex
    - Google Analogies
    """)
    parser.add_argument('emb_file', help="file with word embeddings")
    parser.add_argument('out_file', help="output file")
    parser.add_argument('--pos', type=bool, default=False, help="wether to use data with POS tags")

    args = parser.parse_args()

    if not os.path.isfile(args.emb_file):
        logger.error("embeddings file '%s' does not exist", args.emb_file)
        exit(1)

    logger.info('Loading the embedding model...')

    emb_model = load_model(args.emb_file)
    emb_model.init_sims(replace=True)

    logger.info('Finished loading the embedding model...')
    logger.info('Model vocabulary size: %d' % len(emb_model.wv.vocab))

    results = {
        'analogies': eval_on_analogies(emb_model, logger=logger, pos=args.pos),
        'simlex': eval_on_simlex(emb_model, logger=logger, pos=args.pos)
    }

    logger.info('Saving results to file...')

    with open(args.out_file, "w") as f:
        json.dump(results, f)

    logger.info('Results saved.')
