import argparse
from gensim.models import Word2Vec

parser = argparse.ArgumentParser()
parser.add_argument('-doc', default = "./prepronc.de", help = "the file that save the training data")

opt = parser.parse_args()
docus = [line.rstrip('\n') for line in open(opt.doc)]
texts = [[word for word in document.split()] for document in docus]
model = Word2Vec(texts, size = 500, min_count = 1)
model.save('./german-model')
