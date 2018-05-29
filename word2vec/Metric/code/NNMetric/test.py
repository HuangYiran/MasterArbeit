import argparse
import gensim
import string
import re

import numpy as np
import nltk
#nltk.download('all')
from nltk.tokenize.moses import MosesTokenizer

parser = argparse.ArgumentParser(description='test.py')


parser.add_argument('-ref', required=True,
                    help='Reference file')

parser.add_argument('-hyp', required=True,
                    help='Hypothesis')


parser.add_argument('-w2v', required=True,
                    help='Word2VecModel')

parser.add_argument('-join', default="sum",
                    help='Word2VecModel')


def join(inp,add,op):
    if(op == "sum"):
        return inp + add
    elif(op == "max"):
        return np.maximum(inp,add)
    else:
        print ("Unknown operation:",op)
        exit()

def createVector(line,model,tok,op):
    #punctuation = string.punctuation
    #punctuation = re.sub('@','',punctuation)
    #punctuation = re.sub("'",'',punctuation)
    #f = lambda x: ''.join([i for i in x if i not in punctuation])
    
    sum = np.zeros(300)
    #line = f(line)
    #words = line.split(' ')
    #for w in tok.tokenize(line.strip().decode('utf-8')):
    for w in tok.penn_tokenize(line.strip()):
#    for w in words:
        if w in model.wv.vocab:
            #print (w,model[w].sum())
            sum = join(sum,model[w],op)
        elif w.lower() in model.wv.vocab:
            sum = join(sum,model[w.lower()],op)
        #else:
        #    print ("OOV:",w)
        #    exit()
    return sum
    
def distance(v1,v2):
    return np.absolute((v1 - v2)).sum()

def main():
    opt = parser.parse_args()

    #print ("Reference: 0.1 ",opt.ref,"Hyothesis: ",opt.hyp,"Word2Vec: ",opt.w2v, "0.1")
    tok = MosesTokenizer();

    model = gensim.models.KeyedVectors.load_word2vec_format(opt.w2v,binary=True)
    
    
    rf = open(opt.ref)
    
    hf = open(opt.hyp)
    
    
    rl = rf.readline()
    hl = hf.readline()
    
    while(rl and hl):
        
        rv = createVector(rl,model,tok,opt.join)
        hv = createVector(hl,model,tok,opt.join)
        print ("Distance:", distance(rv,hv));
        
        rl = rf.readline()
        hl = hf.readline()
    
    
if __name__ == "__main__":
    main()
