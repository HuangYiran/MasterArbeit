import sys
sys.path.append('utils/')
import nnXml as xml
import fmin
import Params
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-doc', default = 'fullHiddenModelTestList',
        help = "file to extract the parameter")
parser.add_argument('-getLast', action = 'store_true',
        help = "when set, only get the data of last hidden value")
parser.add_argument('-expr', default = 'plan_a_en.de',
        help = "the name of the experiment, it deside the data of the experiment")

opt = parser.parse_args()
params_list = xml.read_exp_list(opt.doc)

results = []
for params in params_list:
#    params.update(params_data)
    result = fmin.o_func(params)
    results.append(result)
print results
