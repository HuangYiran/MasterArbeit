import argparse
import gensim
import string
import re
import os

import numpy as np
import nltk
#nltk.download('all')
from nltk.tokenize.moses import MosesTokenizer
"""
didn't calculate the distance
only use join to compare the vector if necessary
"""
parser = argparse.ArgumentParser(description='test.py')


#parser.add_argument('-ref', required=True,
#                    help='Reference file')

parser.add_argument('-hyp', required=True,
                    help='Hypothesis')


parser.add_argument('-join', default="sum",
                    help='Word2VecModel')
parser.add_argument('-output', default = '/tmp/tmp1')


def join(inp,add,op):
    if(op == "sum"):
        return inp + add
    elif(op == "max"):
        return np.maximum(inp,add)
    else:
        print ("Unknown operation:",op)
        exit()

def createVector(sentVec,op):
    if len(sentVec.shape) == 1:
        return sentVec
    num_words = sentVec.shape[0]
    sum = np.zeros(500)
    for i in range(num_words):
        sum = join(sum, sentVec[i], op)
    return sum
    
def distance(v1,v2):
    return np.absolute((v1 - v2)).sum()

def main():
    opt = parser.parse_args()

    data_hyp = np.load(opt.hyp)
    len_shape = len(data_hyp.shape)
    if len_shape == 2:
        out = data_hyp
    elif len_shape == 3:
        if opt.join == 'sum':
            out = data_hyp.sum(axis = 1)
        elif opt.join == 'max':
            out = data_hyp.max(axis = 1)
        else:
            print "unreconized join type"
    else:
        print 'the shape of input data is wrong, please check the input data'
    print out.shape
    np.save(opt.output, out)
    
if __name__ == "__main__":
    main()
