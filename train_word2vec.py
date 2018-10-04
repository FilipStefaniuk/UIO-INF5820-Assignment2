import gensim
import logging
import multiprocessing
import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""
        Trains the word embedding model.
    """)
    parser.add_argument('corpus', help='Path to a training corpus')
    parser.add_argument('out', help='Output file where to save model')
    parser.add_argument('--cores', default=False, help='Limit on the number of cores to use')
    parser.add_argument('--sg', type=bool, default=True)
    parser.add_argument('--window', type=int, default=2)
    parser.add_argument('--vocabsize', type=int, default=100000)
    parser.add_argument('--vectorsize', type=int, default=300)
    parser.add_argument('--iterations', type=int, default=2)

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    if not os.path.exists(args.corpus):
        logger.error("Invalid corpus filename")
        exit(1)

    data = gensim.models.word2vec.LineSentence(args.corpus)
    logger.info("Created dataset from corpus '%s'" % args.corpus)

    cores = int(args.cores) if args.cores else multiprocessing.cpu_count()
    logger.info("Number of cores to use: %d" % cores)

    model = gensim.models.Word2Vec(data, size=args.vectorsize, window=args.window, workers=cores, sg=args.sg,
                                   max_final_vocab=args.vocabsize, iter=args.iterations, sample=0)

    logger.info("Saving the model to '%s'...")
    model.wv.save_word2vec_format(args.out, binary=False)
    logger.info("Model saved.")
