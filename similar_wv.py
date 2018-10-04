import os
import json
import argparse
import gensim
import logging
from util import load_model


if __name__ == '__main__':

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="""
    Finds the similar words in embedding model.
    """)
    parser.add_argument('emb_file', help="file with word embeddings")
    parser.add_argument('words_file', help="file with words queries")
    parser.add_argument('--output_file', help="file where to save output")
    parser.add_argument('--low', type=int, default=11, help="low index of similar words")
    parser.add_argument('--high', type=int, default=15, help="high index of similar words")

    args = parser.parse_args()

    if not os.path.isfile(args.emb_file):
        logger.error("embeddings file '%s' does not exist", args.emb_file)
        exit(1)

    if not os.path.isfile(args.words_file):
        logger.error("words file '%s' does not exist", args.words_file)
        exit(1)

    logger.info('Loading the embedding model...')

    emb_model = load_model(args.emb_file)
    emb_model.init_sims(replace=True)

    logger.info('Finished loading the embedding model...')
    logger.info('Model vocabulary size: %d' % len(emb_model.wv.vocab))

    logger.info('Finding similar words...')
    results = []
    with open(args.words_file, "r") as f:

        for line in f:
            word = line.strip()

            if word in emb_model:
                similar = emb_model.wv.most_similar(positive=[word], topn=args.high)
                similar = similar[args.low-1:]
                words, scores = zip(*similar)
            else:
                words, scores = [], []

            results.append({
                'word': word,
                'similar_idx': [args.low + i for i, _ in enumerate(words)],
                'similar_words': words,
                'similar_scores': scores
            })

    logger.info('Finished finding similar words...')

    if args.output_file:

        logger.info('Saving result to the file...')

        with open(args.output_file, "w") as out:
            json.dump(results, out)

        logger.info('Results saved.')

    else:
        print(json.dumps(results))
