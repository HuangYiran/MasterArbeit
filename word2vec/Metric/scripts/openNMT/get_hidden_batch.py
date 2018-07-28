#!-*- coding:utf-8 -*-
import os
"""plan a de
os.system('python get_hidden.py -src ../data/MasterArbeit/plan_a_source_de.en/prepro_extracted_data/preproplan_a_de_sys.bpe.noUndo.de -tgt ../data/MasterArbeit/plan_a_source_de.en/prepro_extracted_data/preproplan_a_de_sys.bpe.noUndo.en -output ../data/MasterArbeit/plan_a_de.en/sys_hidden -get_last')
os.system('python get_hidden.py -src ../data/MasterArbeit/plan_a_source_de.en/prepro_extracted_data/preproplan_a_de_ref.bpe.noUndo.de -tgt ../data/MasterArbeit/plan_a_source_de.en/prepro_extracted_data/preproplan_a_de_ref.bpe.noUndo.en -output ../data/MasterArbeit/plan_a_de.en/ref_hidden -get_last')
"""

"""plan a ed
os.system('python get_hidden.py -src ../data/MasterArbeit/plan_a_source_en.de/prepro_extracted_data/preproplan_a_ed_sys.bpe.noUndo.en -tgt ../data/MasterArbeit/plan_a_source_en.de/prepro_extracted_data/preproplan_a_ed_sys.bpe.noUndo.de -output ../data/MasterArbeit/plan_a_en.de/sys_hidden -get_last')
os.system('python get_hidden.py -src ../data/MasterArbeit/plan_a_source_en.de/prepro_extracted_data/preproplan_a_ed_ref.bpe.noUndo.en -tgt ../data/MasterArbeit/plan_a_source_en.de/prepro_extracted_data/preproplan_a_ed_ref.bpe.noUndo.de -output ../data/MasterArbeit/plan_a_en.de/ref_hidden -get_last')
"""

"""plan b de
os.system('python get_hidden.py -src ../data/MasterArbeit/plan_b_source_de.en/prepro_extracted_data/preproplan_b_de_sys_train.bpe.noUndo.de -tgt ../data/MasterArbeit/plan_b_source_de.en/prepro_extracted_data/preproplan_b_de_sys_train.bpe.noUndo.en -output ../data/MasterArbeit/plan_b_de.en/train_sys_hidden -get_last')
os.system('python get_hidden.py -src ../data/MasterArbeit/plan_b_source_de.en/prepro_extracted_data/preproplan_b_de_ref_train.bpe.noUndo.de -tgt ../data/MasterArbeit/plan_b_source_de.en/prepro_extracted_data/preproplan_b_de_ref_train.bpe.noUndo.en -output ../data/MasterArbeit/plan_b_de.en/train_ref_hidden -get_last')
os.system('python get_hidden.py -src ../data/MasterArbeit/plan_b_source_de.en/prepro_extracted_data/preproplan_b_de_sys_val.bpe.noUndo.de -tgt ../data/MasterArbeit/plan_b_source_de.en/prepro_extracted_data/preproplan_b_de_sys_val.bpe.noUndo.en -output ../data/MasterArbeit/plan_b_de.en/val_sys_hidden -get_last')
os.system('python get_hidden.py -src ../data/MasterArbeit/plan_b_source_de.en/prepro_extracted_data/preproplan_b_de_ref_val.bpe.noUndo.de -tgt ../data/MasterArbeit/plan_b_source_de.en/prepro_extracted_data/preproplan_b_de_ref_val.bpe.noUndo.en -output ../data/MasterArbeit/plan_b_de.en/val_ref_hidden -get_last')
os.system('python get_hidden.py -src ../data/MasterArbeit/plan_b_source_de.en/prepro_extracted_data/preproplan_b_de_sys_test.bpe.noUndo.de -tgt ../data/MasterArbeit/plan_b_source_de.en/prepro_extracted_data/preproplan_b_de_sys_test.bpe.noUndo.en -output ../data/MasterArbeit/plan_b_de.en/test_sys_hidden -get_last')
os.system('python get_hidden.py -src ../data/MasterArbeit/plan_b_source_de.en/prepro_extracted_data/preproplan_b_de_ref_test.bpe.noUndo.de -tgt ../data/MasterArbeit/plan_b_source_de.en/prepro_extracted_data/preproplan_b_de_ref_test.bpe.noUndo.en -output ../data/MasterArbeit/plan_b_de.en/test_ref_hidden -get_last')
"""

"""plan a da de
os.system('python get_hidden.py -src ../data/MasterArbeit/plan_c_source_da_de.en/prepro_extracted_data/preproplan_c_de_sys.bpe.noUndo.de -tgt ../data/MasterArbeit/plan_c_source_da_de.en/prepro_extracted_data/preproplan_c_de_sys.bpe.noUndo.en -output ../data/MasterArbeit/plan_c_da_de.en/sys_hidden_2016 -get_last')
os.system('python get_hidden.py -src ../data/MasterArbeit/plan_c_source_da_de.en/prepro_extracted_data/preproplan_c_de_ref.bpe.noUndo.de -tgt ../data/MasterArbeit/plan_c_source_da_de.en/prepro_extracted_data/preproplan_c_de_ref.bpe.noUndo.en -output ../data/MasterArbeit/plan_c_da_de.en/ref_hidden_2016 -get_last')
"""
"""
#plan c rank ed, 别忘了使用ed模型
os.system('python get_hidden.py -src ../data/MasterArbeit/plan_c_source_rank_en.de/prepro_extracted_data/preproplan_c_rank_ende_training_ref_2017.bpe.noUndo.en -tgt ../data/MasterArbeit/plan_c_source_rank_en.de/prepro_extracted_data/preproplan_c_rank_ende_training_ref_2017.bpe.noUndo.de -output ../data/MasterArbeit/plan_c_rank_en.de/hidden_traing_ref -get_last')
os.system('python get_hidden.py -src ../data/MasterArbeit/plan_c_source_rank_en.de/prepro_extracted_data/preproplan_c_rank_ende_training_s1_2017.bpe.noUndo.en -tgt ../data/MasterArbeit/plan_c_source_rank_en.de/prepro_extracted_data/preproplan_c_rank_ende_training_s1_2017.bpe.noUndo.de -output ../data/MasterArbeit/plan_c_rank_en.de/hidden_training_s1 -get_last')
os.system('python get_hidden.py -src ../data/MasterArbeit/plan_c_source_rank_en.de/prepro_extracted_data/preproplan_c_rank_ende_training_s2_2017.bpe.noUndo.en -tgt ../data/MasterArbeit/plan_c_source_rank_en.de/prepro_extracted_data/preproplan_c_rank_ende_training_s2_2017.bpe.noUndo.de -output ../data/MasterArbeit/plan_c_rank_en.de/hidden_training_s2 -get_last')
os.system('python get_hidden.py -src ../data/MasterArbeit/plan_c_source_rank_en.de/prepro_extracted_data/preproplan_c_rank_ende_testing_ref_2017.bpe.noUndo.en -tgt ../data/MasterArbeit/plan_c_source_rank_en.de/prepro_extracted_data/preproplan_c_rank_ende_testing_ref_2017.bpe.noUndo.de -output ../data/MasterArbeit/plan_c_rank_en.de/hidden_testing_ref -get_last')
os.system('python get_hidden.py -src ../data/MasterArbeit/plan_c_source_rank_en.de/prepro_extracted_data/preproplan_c_rank_ende_testing_s1_2017.bpe.noUndo.en -tgt ../data/MasterArbeit/plan_c_source_rank_en.de/prepro_extracted_data/preproplan_c_rank_ende_testing_s1_2017.bpe.noUndo.de -output ../data/MasterArbeit/plan_c_rank_en.de/hidden_testing_s1 -get_last')
os.system('python get_hidden.py -src ../data/MasterArbeit/plan_c_source_rank_en.de/prepro_extracted_data/preproplan_c_rank_ende_testing_s2_2017.bpe.noUndo.en -tgt ../data/MasterArbeit/plan_c_source_rank_en.de/prepro_extracted_data/preproplan_c_rank_ende_testing_s2_2017.bpe.noUndo.de -output ../data/MasterArbeit/plan_c_rank_en.de/hidden_testing_s2 -get_last')
"""

"""
#plan c rank de 别忘了使用de模型
os.system('python get_hidden.py -src ../data/MasterArbeit/plan_c_source_rank_de.en/prepro_extracted_data/preproplan_c_de_1415_ref.bpe.noUndo.de -tgt ../data/MasterArbeit/plan_c_source_rank_de.en/prepro_extracted_data/preproplan_c_de_1415_ref.bpe.noUndo.en -output ../data/MasterArbeit/plan_c_rank_de.en/hidden_de_1415_ref -get_last')
os.system('python get_hidden.py -src ../data/MasterArbeit/plan_c_source_rank_de.en/prepro_extracted_data/preproplan_c_de_1415_s1.bpe.noUndo.de -tgt ../data/MasterArbeit/plan_c_source_rank_de.en/prepro_extracted_data/preproplan_c_de_1415_s1.bpe.noUndo.en -output ../data/MasterArbeit/plan_c_rank_de.en/hidden_de_1415_s1 -get_last')
os.system('python get_hidden.py -src ../data/MasterArbeit/plan_c_source_rank_de.en/prepro_extracted_data/preproplan_c_de_1415_s2.bpe.noUndo.de -tgt ../data/MasterArbeit/plan_c_source_rank_de.en/prepro_extracted_data/preproplan_c_de_1415_s2.bpe.noUndo.en -output ../data/MasterArbeit/plan_c_rank_de.en/hidden_de_1415_s2 -get_last')
os.system('python get_hidden.py -src ../data/MasterArbeit/plan_c_source_rank_de.en/prepro_extracted_data/preproplan_c_de_2016_ref.bpe.noUndo.de -tgt ../data/MasterArbeit/plan_c_source_rank_de.en/prepro_extracted_data/preproplan_c_de_2016_ref.bpe.noUndo.en -output ../data/MasterArbeit/plan_c_rank_de.en/hidden_de_2016_ref -get_last')
os.system('python get_hidden.py -src ../data/MasterArbeit/plan_c_source_rank_de.en/prepro_extracted_data/preproplan_c_de_2016_s1.bpe.noUndo.de -tgt ../data/MasterArbeit/plan_c_source_rank_de.en/prepro_extracted_data/preproplan_c_de_2016_s1.bpe.noUndo.en -output ../data/MasterArbeit/plan_c_rank_de.en/hidden_de_2016_s1 -get_last')
os.system('python get_hidden.py -src ../data/MasterArbeit/plan_c_source_rank_de.en/prepro_extracted_data/preproplan_c_de_2016_s2.bpe.noUndo.de -tgt ../data/MasterArbeit/plan_c_source_rank_de.en/prepro_extracted_data/preproplan_c_de_2016_s2.bpe.noUndo.en -output ../data/MasterArbeit/plan_c_rank_de.en/hidden_de_2016_s2 -get_last')
"""
"""
# plan c da de
os.system('python get_hidden.py -src ../data/MasterArbeit/plan_c_source_da_de.en/prepro_extracted_sub_human/data_src -tgt ../data/MasterArbeit/plan_c_source_da_de.en/prepro_extracted_sub_human/data_ref -output ../data/MasterArbeit/plan_c_da_de.en/sub_accuracy/human/hidden_training_ref -get_last')
os.system('python get_hidden.py -src ../data/MasterArbeit/plan_c_source_da_de.en/prepro_extracted_sub_metrics/data_src -tgt ../data/MasterArbeit/plan_c_source_da_de.en/prepro_extracted_sub_metrics/data_ref -output ../data/MasterArbeit/plan_c_da_de.en/sub_accuracy/metrics/hidden_training_ref -get_last')
os.system('python get_hidden.py -src ../data/MasterArbeit/plan_c_source_da_de.en/prepro_extracted_sub_human/data_src -tgt ../data/MasterArbeit/plan_c_source_da_de.en/prepro_extracted_sub_human/data_sys -output ../data/MasterArbeit/plan_c_da_de.en/sub_accuracy/human/hidden_training_sys -get_last')
os.system('python get_hidden.py -src ../data/MasterArbeit/plan_c_source_da_de.en/prepro_extracted_sub_metrics/data_src -tgt ../data/MasterArbeit/plan_c_source_da_de.en/prepro_extracted_sub_metrics/data_sys -output ../data/MasterArbeit/plan_c_da_de.en/sub_accuracy/metrics/hidden_training_sys -get_last')
"""
#plan c rank deen
print '#########################'
print '# plan c rank deen'
print '#########################'

print '+++++++++++13+++++++++++'
os.system('python get_hidden.py -model ../data/mt_model/model_deen -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2013/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2013/data_ref -output ../data/MasterArbeit/plan_c_rank_de.en/13/hidden_de_13_ref -get_last')
os.system('python get_hidden.py -model ../data/mt_model/model_deen -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2013/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2013/data_s1 -output ../data/MasterArbeit/plan_c_rank_de.en/13/hidden_de_13_s1 -get_last')
os.system('python get_hidden.py -model ../data/mt_model/model_deen -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2013/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2013/data_s2 -output ../data/MasterArbeit/plan_c_rank_de.en/13/hidden_de_13_s2 -get_last')
os.system('cp /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2013/data_scores ../data/MasterArbeit/plan_c_rank_de.en/13/')

print '+++++++++++14+++++++++++'
os.system('python get_hidden.py -model ../data/mt_model/model_deen -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2014/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2014/data_ref -output ../data/MasterArbeit/plan_c_rank_de.en/14/hidden_de_14_ref -get_last')
os.system('python get_hidden.py -model ../data/mt_model/model_deen -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2014/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2014/data_s1 -output ../data/MasterArbeit/plan_c_rank_de.en/14/hidden_de_14_s1 -get_last')
os.system('python get_hidden.py -model ../data/mt_model/model_deen -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2014/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2014/data_s2 -output ../data/MasterArbeit/plan_c_rank_de.en/14/hidden_de_14_s2 -get_last')
os.system('cp /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2014/data_scores ../data/MasterArbeit/plan_c_rank_de.en/14/')

print '++++++++++++15+++++++++++'
os.system('python get_hidden.py -model ../data/mt_model/model_deen -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_ref -output ../data/MasterArbeit/plan_c_rank_de.en/15/hidden_de_15_ref -get_last')
os.system('python get_hidden.py -model ../data/mt_model/model_deen -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s1 -output ../data/MasterArbeit/plan_c_rank_de.en/15/hidden_de_15_s1 -get_last')
os.system('python get_hidden.py -model ../data/mt_model/model_deen -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s2 -output ../data/MasterArbeit/plan_c_rank_de.en/15/hidden_de_15_s2 -get_last')
os.system('cp /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2015/data_scores ../data/MasterArbeit/plan_c_rank_de.en/15/')

print '+++++++++++++16+++++++++++'
os.system('python get_hidden.py -model ../data/mt_model/model_deen -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_ref -output ../data/MasterArbeit/plan_c_rank_de.en/16/hidden_de_16_ref -get_last')
os.system('python get_hidden.py -model ../data/mt_model/model_deen -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s1 -output ../data/MasterArbeit/plan_c_rank_de.en/16/hidden_de_16_s1 -get_last')
os.system('python get_hidden.py -model ../data/mt_model/model_deen -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s2 -output ../data/MasterArbeit/plan_c_rank_de.en/16/hidden_de_16_s2 -get_last')
os.system('cp /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2016/data_scores ../data/MasterArbeit/plan_c_rank_de.en/16/')


# plan c rank ende
print '#########################'
print '# plan c rank ende'
print '#########################'

print '++++++++++++++13+++++++++++++'
os.system('python get_hidden.py -model ../data/mt_model/model_ende -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2013/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2013/data_ref -output ../data/MasterArbeit/plan_c_rank_en.de/13/data_ref -get_last')
os.system('python get_hidden.py -model ../data/mt_model/model_ende -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2013/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2013/data_s1 -output ../data/MasterArbeit/plan_c_rank_en.de/13/data_s1 -get_last')
os.system('python get_hidden.py -model ../data/mt_model/model_ende -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2013/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2013/data_s2 -output ../data/MasterArbeit/plan_c_rank_en.de/13/data_s2 -get_last')
os.system('cp /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2013/data_scores ../data/MasterArbeit/plan_c_rank_en.de/13/')

print '+++++++++++++14+++++++++++++'
os.system('python get_hidden.py -model ../data/mt_model/model_ende -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2014/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2014/data_ref -output ../data/MasterArbeit/plan_c_rank_en.de/14/data_ref -get_last')
os.system('python get_hidden.py -model ../data/mt_model/model_ende -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2014/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2014/data_s1 -output ../data/MasterArbeit/plan_c_rank_en.de/14/data_s1 -get_last')
os.system('python get_hidden.py -model ../data/mt_model/model_ende -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2014/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2014/data_s2 -output ../data/MasterArbeit/plan_c_rank_en.de/14/data_s2 -get_last')
os.system('cp /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2014/data_scores ../data/MasterArbeit/plan_c_rank_en.de/14/')

print '++++++++++++++15+++++++++++++'
os.system('python get_hidden.py -model ../data/mt_model/model_ende -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_ref -output ../data/MasterArbeit/plan_c_rank_en.de/15/data_ref -get_last')
os.system('python get_hidden.py -model ../data/mt_model/model_ende -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_s1 -output ../data/MasterArbeit/plan_c_rank_en.de/15/data_s1 -get_last')
os.system('python get_hidden.py -model ../data/mt_model/model_ende -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_s2 -output ../data/MasterArbeit/plan_c_rank_en.de/15/data_s2 -get_last')
os.system('cp /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2015/data_scores ../data/MasterArbeit/plan_c_rank_en.de/15/')

print '++++++++++++++16++++++++++++++'
os.system('python get_hidden.py -model ../data/mt_model/model_ende -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_ref -output ../data/MasterArbeit/plan_c_rank_en.de/16/data_ref -get_last')
os.system('python get_hidden.py -model ../data/mt_model/model_ende -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_s1 -output ../data/MasterArbeit/plan_c_rank_en.de/16/data_s1 -get_last')
os.system('python get_hidden.py -model ../data/mt_model/model_ende -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_s2 -output ../data/MasterArbeit/plan_c_rank_en.de/16/data_s2 -get_last')
os.system('cp /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2016/data_scores ../data/MasterArbeit/plan_c_rank_en.de/16/')






