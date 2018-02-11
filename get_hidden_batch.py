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

"""plan a da de"""
os.system('python get_hidden.py -src ../data/MasterArbeit/plan_c_source_da_de.en/prepro_extracted_data/preproplan_c_de_sys.bpe.noUndo.de -tgt ../data/MasterArbeit/plan_c_source_da_de.en/prepro_extracted_data/preproplan_c_de_sys.bpe.noUndo.en -output ../data/MasterArbeit/plan_c_da_de.en/sys_hidden_2016 -get_last')
os.system('python get_hidden.py -src ../data/MasterArbeit/plan_c_source_da_de.en/prepro_extracted_data/preproplan_c_de_ref.bpe.noUndo.de -tgt ../data/MasterArbeit/plan_c_source_da_de.en/prepro_extracted_data/preproplan_c_de_ref.bpe.noUndo.en -output ../data/MasterArbeit/plan_c_da_de.en/ref_hidden_2016 -get_last')


"""plan c rank ed
os.system('python get_hidden.py -src ../data/MasterArbeit/plan_c_source_rank_en.de/prepro_extracted_data/preproplan_c_ed_2014_ref.bpe.noUndo.en -tgt ../data/MasterArbeit/plan_c_source_rank_en.de/prepro_extracted_data/preproplan_c_ed_2014_ref.bpe.noUndo.de -output ../data/MasterArbeit/plan_c_rank_en.de/hidden_ed_2014_ref -get_last')
os.system('python get_hidden.py -src ../data/MasterArbeit/plan_c_source_rank_en.de/prepro_extracted_data/preproplan_c_ed_2014_s1.bpe.noUndo.en -tgt ../data/MasterArbeit/plan_c_source_rank_en.de/prepro_extracted_data/preproplan_c_ed_2014_s1.bpe.noUndo.de -output ../data/MasterArbeit/plan_c_rank_en.de/hidden_ed_2014_s1 -get_last')
os.system('python get_hidden.py -src ../data/MasterArbeit/plan_c_source_rank_en.de/prepro_extracted_data/preproplan_c_ed_2014_s2.bpe.noUndo.en -tgt ../data/MasterArbeit/plan_c_source_rank_en.de/prepro_extracted_data/preproplan_c_ed_2014_s2.bpe.noUndo.de -output ../data/MasterArbeit/plan_c_rank_en.de/hidden_ed_2014_s2 -get_last')
os.system('python get_hidden.py -src ../data/MasterArbeit/plan_c_source_rank_en.de/prepro_extracted_data/preproplan_c_ed_2015_ref.bpe.noUndo.en -tgt ../data/MasterArbeit/plan_c_source_rank_en.de/prepro_extracted_data/preproplan_c_ed_2015_ref.bpe.noUndo.de -output ../data/MasterArbeit/plan_c_rank_en.de/hidden_ed_2015_ref -get_last')
os.system('python get_hidden.py -src ../data/MasterArbeit/plan_c_source_rank_en.de/prepro_extracted_data/preproplan_c_ed_2015_s1.bpe.noUndo.en -tgt ../data/MasterArbeit/plan_c_source_rank_en.de/prepro_extracted_data/preproplan_c_ed_2015_s1.bpe.noUndo.de -output ../data/MasterArbeit/plan_c_rank_en.de/hidden_ed_2015_s1 -get_last')
os.system('python get_hidden.py -src ../data/MasterArbeit/plan_c_source_rank_en.de/prepro_extracted_data/preproplan_c_ed_2015_s2.bpe.noUndo.en -tgt ../data/MasterArbeit/plan_c_source_rank_en.de/prepro_extracted_data/preproplan_c_ed_2015_s2.bpe.noUndo.de -output ../data/MasterArbeit/plan_c_rank_en.de/hidden_ed_2015_s2 -get_last')
"""

"""plan c rank de
os.system('python get_hidden.py -src ../data/MasterArbeit/plan_c_source_rank_de.en/prepro_extracted_data/preproplan_c_de_2014_ref.bpe.noUndo.de -tgt ../data/MasterArbeit/plan_c_source_rank_de.en/prepro_extracted_data/preproplan_c_de_2014_ref.bpe.noUndo.en -output ../data/MasterArbeit/plan_c_rank_de.en/hidden_de_2014_ref -get_last')
os.system('python get_hidden.py -src ../data/MasterArbeit/plan_c_source_rank_de.en/prepro_extracted_data/preproplan_c_de_2014_s1.bpe.noUndo.de -tgt ../data/MasterArbeit/plan_c_source_rank_de.en/prepro_extracted_data/preproplan_c_de_2014_s1.bpe.noUndo.en -output ../data/MasterArbeit/plan_c_rank_de.en/hidden_de_2014_s1 -get_last')
os.system('python get_hidden.py -src ../data/MasterArbeit/plan_c_source_rank_de.en/prepro_extracted_data/preproplan_c_de_2014_s2.bpe.noUndo.de -tgt ../data/MasterArbeit/plan_c_source_rank_de.en/prepro_extracted_data/preproplan_c_de_2014_s2.bpe.noUndo.en -output ../data/MasterArbeit/plan_c_rank_de.en/hidden_de_2014_s2 -get_last')
os.system('python get_hidden.py -src ../data/MasterArbeit/plan_c_source_rank_de.en/prepro_extracted_data/preproplan_c_de_2015_ref.bpe.noUndo.de -tgt ../data/MasterArbeit/plan_c_source_rank_de.en/prepro_extracted_data/preproplan_c_de_2015_ref.bpe.noUndo.en -output ../data/MasterArbeit/plan_c_rank_de.en/hidden_de_2015_ref -get_last')
os.system('python get_hidden.py -src ../data/MasterArbeit/plan_c_source_rank_de.en/prepro_extracted_data/preproplan_c_de_2015_s1.bpe.noUndo.de -tgt ../data/MasterArbeit/plan_c_source_rank_de.en/prepro_extracted_data/preproplan_c_de_2015_s1.bpe.noUndo.en -output ../data/MasterArbeit/plan_c_rank_de.en/hidden_de_2015_s1 -get_last')
os.system('python get_hidden.py -src ../data/MasterArbeit/plan_c_source_rank_de.en/prepro_extracted_data/preproplan_c_de_2015_s2.bpe.noUndo.de -tgt ../data/MasterArbeit/plan_c_source_rank_de.en/prepro_extracted_data/preproplan_c_de_2015_s2.bpe.noUndo.en -output ../data/MasterArbeit/plan_c_rank_de.en/hidden_de_2015_s2 -get_last')
"""
