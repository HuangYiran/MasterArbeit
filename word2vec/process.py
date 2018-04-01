from get_hidden import get_hidden_with_word2vec
from gensim.models import Word2Vec
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-in_dir', default = './raw_data/deen/2015/', help = "directory, under which include the file that want to be translated")
parser.add_argument('-out_dir', default = './hidden_data/deen/')
parser.add_argument('-model', default = './english/english-model', help = "the word2vec model, should coordinate with the input file")

opt = parser.parse_args()
# read the model
model = Word2Vec.load(opt.model)

f1 = './raw_data/deen/2015/data_ref'
f2 = './raw_data/deen/2015/data_sys'
f3 = './raw_data/deen/2016/data_ref'
f4 = './raw_data/deen/2016/data_sys'
f5 = './raw_data/deen/2017/data_ref'
f6 = './raw_data/deen/2017/data_sys'

# get hidden for the data
h_2015_ref = get_hidden_with_word2vec(f1, model)
h_2015_sys = get_hidden_with_word2vec(f2, model)
h_2016_ref = get_hidden_with_word2vec(f3, model)
h_2016_sys = get_hidden_with_word2vec(f4, model)
h_2017_ref = get_hidden_with_word2vec(f5, model)
h_2017_sys = get_hidden_with_word2vec(f6, model)

h_train_ref = h_2015_ref
h_train_ref.extend(h_2016_ref)
h_train_sys = h_2015_sys
h_train_sys.extend(h_2016_sys)
h_test_ref = h_2017_ref
h_test_sys = h_2017_sys

h_train_ref = np.asarray(h_train_ref)
h_train_sys = np.asarray(h_train_sys)
h_test_ref = np.asarray(h_test_ref)
h_test_sys = np.asarray(h_test_sys)

# write the data
np.save('./hidden_data/deen/train_ref', h_train_ref)
np.save('./hidden_data/deen/train_sys', h_train_sys)
np.save('./hidden_data/deen/test_ref', h_test_ref)
np.save('./hidden_data/deen/test_sys', h_test_sys)

"""
filenames = os.listdir(opt.in_dir)
for filename in filenames:
    print 'processing the file '+ filename
    out_dir = opt.out_dir+"hidden_"+filename
    h_doc = get_hidden_with_word2vec(opt.in_dir+filename, model)
    print 'save the file to the dir ' + out_dir
    np.save(out_dir, h_doc)
"""
