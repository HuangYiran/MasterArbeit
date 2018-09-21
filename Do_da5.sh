"""
aborted!!
Do_da.sh is the code to get the embedding
Do_da2.sh is the code to train the model but hold the distance function for direct distance based model
Do_da3.sh is the code to train the model but hold the aggregation function for direct distance based model
Do_da4.sh is the code to train the model but hold the distance function for mapped distance based model
Do_da5.sh is the code to train the model but hold the aggregation function for mapped distance based model
"""

echo '######################'
echo '### mepper ###########'
echo '######################'
echo 'with the order, max-mean-sum'
echo '***** w2v *****'
python experiments/train_model_da_mepper.py -tgt_s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_da_de.en/w2v/w2v_deen_2016_sys_max.npy -tgt_ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_da_de.en/w2v/w2v_deen_2016_ref_max.npy -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_da_de.en/data_scores_2016 -test_s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_da_de.en/w2v/w2v_deen_2017_sys_max.npy -test_ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_da_de.en/w2v/w2v_deen_2017_ref_max.npy -test_scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_da_de.en/data_scores_2017 
python experiments/train_model_da_mepper.py -tgt_s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_da_de.en/w2v/w2v_deen_2016_sys_max.npy -tgt_ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_da_de.en/w2v/w2v_deen_2016_ref_max.npy -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_da_de.en/data_scores_2016 -test_s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_da_de.en/w2v/w2v_deen_2017_sys_max.npy -test_ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_da_de.en/w2v/w2v_deen_2017_ref_max.npy -test_scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_da_de.en/data_scores_2017 
python experiments/train_model_da_mepper.py -tgt_s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_da_de.en/w2v/w2v_deen_2016_sys_max.npy -tgt_ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_da_de.en/w2v/w2v_deen_2016_ref_max.npy -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_da_de.en/data_scores_2016 -test_s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_da_de.en/w2v/w2v_deen_2017_sys_max.npy -test_ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_da_de.en/w2v/w2v_deen_2017_ref_max.npy -test_scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_da_de.en/data_scores_2017 
python experiments/train_model_da_mepper.py -tgt_s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_da_de.en/w2v/w2v_deen_2016_sys_max.npy -tgt_ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_da_de.en/w2v/w2v_deen_2016_ref_max.npy -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_da_de.en/data_scores_2016 -test_s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_da_de.en/w2v/w2v_deen_2017_sys_max.npy -test_ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_da_de.en/w2v/w2v_deen_2017_ref_max.npy -test_scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_da_de.en/data_scores_2017 

