import argparse
import os
from gensim.models import Word2Vec

parser = argparse.ArgumentParser()
parser.add_argument('-doc', default = "./prepronc.en", help = "the file that save the training data")
parser.add_argument('-extrac_data', default = "./data/", help = "the extra data that are used to train the model")

opt = parser.parse_args()
docus = [line.rstrip('\n') for line in open(opt.doc)]
texts = [[word for word in document.split()] for document in docus]
model = Word2Vec(texts, size = 500, min_count = 0)

model.save('./english-model')
