import os
import sys
import logging
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing
from tqdm import tqdm
def word2vec_training(file,parameters):
    # For logging.
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    
    # The input and output file.

    for vector_size,windows_size in tqdm(parameters):
        outp1 = 'word2vec_%d_%d_model'%(windows_size,vector_size)
        outp2 = 'word2vec_%d_%d_vector'%(windows_size,vector_size)
        #skip-gram
        # Training a word2vec model using the input file.
        model = Word2Vec( LineSentence(file),epochs=20,vector_size = vector_size,window=windows_size, min_count=5, sg=1,hs =1,workers=multiprocessing.cpu_count() )
        model.save(outp1)
        model.wv.save_word2vec_format(outp2, binary=False)