import sys
sys.path.append('utils/')
import nnXml as xml
import fmin
import Params
import argparse
import numpy as np

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
o = open('/tmp/cv_result', 'w')
for params in params_list:
#    params.update(params_data)
    tmp = [0,0,0]
    for i in range(10):
        assert(params['cross_val'] == True)
        params['cv_val_index'] = i
        result = fmin.o_func(params)
        o.write(str(result))
        tmp = np.add(tmp, result)
    #map(lambda x: x * 0,1, tmp)
    tmp = tmp * 0.1
    results.append(tmp)
print results
