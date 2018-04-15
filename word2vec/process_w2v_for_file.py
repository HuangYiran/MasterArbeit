"""
wrap the get_hidden function to get the vector for each sentence
This is the general version, you can use the parameters to set the input, output and model 
"""
from get_hidden import get_hidden_with_word2vec
from gensim.models import Word2Vec
import gensim
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-src', default = './raw_data/rr/deen/1314/data_sys', help= 'the input file, which store the sentences')
parser.add_argument('-out', default = './hidden_data/rr/deen/1314/data_sys', help = 'the output file to store the result')
parser.add_argument('-model', default = './models/english/GoogleNews-vectors-negative300.bin', help = 'the word2vec model, should coordinate with the input file')
parser.add_argument('--bin', action = 'store_true', help = 'weather the model is binary or not')

def main():
    opt = parser.parse_args()
    print '>>>transform data from the file '+str(opt.src)
    # read the model
    if opt.bin:
        #model = Word2Vec.load_word2vec_format(opt.model, binary = True)
        model = gensim.models.KeyedVectors.load_word2vec_format(opt.model, binary = True)
    else:
        model = Word2Vec.load(opt.model)
    # read data and get the vectors
    h_in = get_hidden_with_word2vec(opt.src, model)
    h_in = np.asarray(h_in)
    # write the data
    np.save(opt.out, h_in)
    print '<<< end '

if __name__ == '__main__':
    main()
