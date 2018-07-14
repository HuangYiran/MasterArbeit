#-*- coding: UTF-8 -*-

import numpy as np
import random
import os
import sys
sys.path.append('./utils')

from collections import namedtuple

class Greedy:
    def run(self, num_fs = 18):
        candidate = []
        Evaluate = namedtuple('Evaluate', ['candidate', 'score'])
        best_of_all = Evaluate(candidate = 0, score = 0)
        for i in range(num_fs):
            best_in_round = Evaluate(candidate = 0, score = 0)
            for j in range(num_fs):
                # choose a feature and add it to the candidate
                tmp = candidate
                tmp.append(j)
                str_tmp = self._intlist_to_string(tmp)
                print('#'*50)
                print('candidate: {}'.format(tmp))
                print('#'*50)
                # train the model with features in tmp
                os.system('python experiments/train_model.py -tgt_s1 /tmp/decMixture_2015_s1 -tgt_s2 /tmp/decMixture_2015_s2 -tgt_ref /tmp/decMixture_2015_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2015/data_scores -cand ' + str_tmp )
                # test the model with features in tmp and save the result in /tmp/testTaul
                os.system('python experiments/test_model.py -tgt_s1 /tmp/decMixture_2016_s1 -tgt_s2 /tmp/decMixture_2016_s2 -tgt_ref /tmp/decMixture_2016_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2016/data_scores -checkpoint /tmp/decState_params > /tmp/testTaul')
                # read the result
                test_taul = [float(li.rstrip('\n').split(' ')) for li in open('/tmp/testTaul')][0]
                if test_taul > best_in_round.score:
                    # choose the best
                    best_in_round = Evaluate(candidate = tmp, score = test_taul)
            # add the best to the candidate list
            candidate.append(best_in_round.candidate)
            # update the best_of_all tuple
            if best_in_round.score > best_of_all.score:
                best_of_all = Evaluate(candidate = best_in_round.candidate, score = best_in_round.score)
            np.save('/tmp/best_in_round', best_in_round)
            np.save('/tmp/best_of_all', best_of_all)
            print('best record: \ncandicate: {} taul: {}'.format(best_of_all.candidate, best_of_all.score))

    def _intlist_to_string(self, li):
        tmp = [str(i) for i in li]
        return ' '.join(tmp)
