import fasttext_training
import itertools
import word2vec_training
import glove_training




if __name__ == '__main__':
    file = 'Mecabtext.txt'
    vector_size = [100,200,300,400,500]
    windows_size = [10,15,20]
    parameters = list(itertools.product(vector_size,windows_size))
    for vector_size,windows_size in parameters:
        word2vec_training.word2vec_training(file,parameters)
        glove_training.glove_training(file,parameters)
        fasttext_training.fasttext_training(file,parameters)
    