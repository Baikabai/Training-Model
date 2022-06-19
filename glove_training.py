from glove import Glove, corpus
from glove import Corpus
from gensim.models.word2vec import LineSentence
def Glove_training(file,parameters):
    for vector_size,windows_size in parameters:
    # corpus training
        sentense = LineSentence(file)
        corpus_model = Corpus()
        corpus_model.fit(sentense, window=windows_size)
        corpus_model.save('corpus.model')
        # glove training
        corpus_model = Corpus.load('corpus.model')
        glove = Glove(no_components=vector_size, learning_rate=0.05)
        glove.fit(corpus_model.matrix, epochs=30, no_threads=1, verbose=True)
        glove.add_dictionary(corpus_model.dictionary)
        glove.save('glove_%d_%d.model'%(windows_size,vector_size))