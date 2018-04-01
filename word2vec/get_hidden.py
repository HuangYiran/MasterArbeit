import numpy as np
from gensim.models import Word2Vec

def get_hidden_with_word2vec(in_file, model):
    sents = [line.rstrip('\n') for line in open(in_file)]
    h_doc = []
    for sent in sents:
        words = sent.split(' ')
        h_sent = model[words.pop(0)]
        h_sent.flags['WRITEABLE'] = True
        for word in words:
            if len(word) == 0:
                continue
            tmp = model[word]
            tmp.flags['WRITEABLE'] = True
            h_sent += tmp 
        h_doc.append(h_sent)
    #h_doc = np.asarray(h_doc)
    print len(h_doc)
    return h_doc
