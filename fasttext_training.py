import os
import sys
import logging
import multiprocessing
from gensim.models import FastText
from gensim.models.word2vec import LineSentence

def fasttext_training(file,parameters):
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    for vector_size,windows_size in parameters:
        
        file = 'C:/Users/umiko/Desktop/M2/reproduction paper/wiki40_ja/finaldata/test_0.txt'
        outp1 = 'fasttext_%d_%d_model'%(windows_size,vector_size)
        outp2 = 'fasttext_%d_%d_vector'%(windows_size,vector_size)
        model = FastText(
            LineSentence(file),
            vector_size=vector_size,
            window=windows_size,
            min_count=5,
            sg=1,
            hs =1,
            epochs=10,
            min_n=3,
            max_n=6,
            workers=multiprocessing.cpu_count()
            )
        
        model.save(out1)
        model.wv.save_word2vec_format(outp2, binary=False)

        