"""
Do_rr2: script for distance based model direct model: train_model_rr_distance 
Do_rr3: script for distance based model mapped model: Simple6
Do_rr4: script for neural based model 1 output model: Simple9
Do_rr5: script for neural based model 3 output model: train_model-Simple0
Do_rr6: script for neural based model half model: Simple1
Do_rr7: script for neural based model mul model: Simpoe2
"""


echo '######################'
echo '### direct ###########'
echo '######################'
echo 'with the order, max-mean-sum'
echo '***** w2v *****'
python experiments/train_model_rr_distance.py -tgt_s1 /tmp/w2v_deen_2016_s1_max.npy -tgt_s2 /tmp/w2v_deen_2016_s2_max.npy -tgt_ref /tmp/w2v_deen_2016_ref_max.npy -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_rank_de.en/data_scores_2016 
python experiments/train_model_rr_distance.py -tgt_s1 /tmp/w2v_deen_2016_s1_mean.npy -tgt_s2 /tmp/w2v_deen_2016_s2_mean.npy -tgt_ref /tmp/w2v_deen_2016_ref_mean.npy -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_rank_de.en/data_scores_2016 
python experiments/train_model_rr_distance.py -tgt_s1 /tmp/w2v_deen_2016_s1_sum.npy -tgt_s2 /tmp/w2v_deen_2016_s2_sum.npy -tgt_ref /tmp/w2v_deen_2016_ref_sum.npy -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_rank_de.en/data_scores_2016 
echo '***** w2v *****'
python experiments/train_model_rr_distance.py -tgt_s1 /tmp/w2v_slim_deen_2016_s1_max.npy -tgt_s2 /tmp/w2v_slim_deen_2016_s2_max.npy -tgt_ref /tmp/w2v_slim_deen_2016_ref_max.npy -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_rank_de.en/data_scores_2016 
python experiments/train_model_rr_distance.py -tgt_s1 /tmp/w2v_slim_deen_2016_s1_mean.npy -tgt_s2 /tmp/w2v_slim_deen_2016_s2_mean.npy -tgt_ref /tmp/w2v_slim_deen_2016_ref_mean.npy -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_rank_de.en/data_scores_2016 
python experiments/train_model_rr_distance.py -tgt_s1 /tmp/w2v_slim_deen_2016_s1_sum.npy -tgt_s2 /tmp/w2v_slim_deen_2016_s2_sum.npy -tgt_ref /tmp/w2v_slim_deen_2016_ref_sum.npy -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_rank_de.en/data_scores_2016 
echo '***** encoderEmbd *****'
python experiments/train_model_rr_distance.py -tgt_s1 /tmp/encoderEmb_deen_2016_mix_s1 -tgt_s2 /tmp/encoderEmb_deen_2016_mix_s2 -tgt_ref /tmp/encoderEmb_deen_2016_mix_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_rank_de.en/data_scores_2016 -cand 1
python experiments/train_model_rr_distance.py -tgt_s1 /tmp/encoderEmb_deen_2016_mix_s1 -tgt_s2 /tmp/encoderEmb_deen_2016_mix_s2 -tgt_ref /tmp/encoderEmb_deen_2016_mix_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_rank_de.en/data_scores_2016 -cand 2
python experiments/train_model_rr_distance.py -tgt_s1 /tmp/encoderEmb_deen_2016_mix_s1 -tgt_s2 /tmp/encoderEmb_deen_2016_mix_s2 -tgt_ref /tmp/encoderEmb_deen_2016_mix_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_rank_de.en/data_scores_2016 -cand 0
echo '***** decoderEmbd *****'
python experiments/train_model_rr_distance.py -tgt_s1 /tmp/decMixture_deen_2016_s1 -tgt_s2 /tmp/decMixture_deen_2016_s2 -tgt_ref /tmp/decMixture_deen_2016_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_rank_de.en/data_scores_2016 -cand 1
python experiments/train_model_rr_distance.py -tgt_s1 /tmp/decMixture_deen_2016_s1 -tgt_s2 /tmp/decMixture_deen_2016_s2 -tgt_ref /tmp/decMixture_deen_2016_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_rank_de.en/data_scores_2016 -cand 2
python experiments/train_model_rr_distance.py -tgt_s1 /tmp/decMixture_deen_2016_s1 -tgt_s2 /tmp/decMixture_deen_2016_s2 -tgt_ref /tmp/decMixture_deen_2016_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_rank_de.en/data_scores_2016 -cand 0
echo '***** decoderOut *****'
python experiments/train_model_rr_distance.py -tgt_s1 /tmp/decMixture_deen_2016_s1 -tgt_s2 /tmp/decMixture_deen_2016_s2 -tgt_ref /tmp/decMixture_deen_2016_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_rank_de.en/data_scores_2016 -cand 4
python experiments/train_model_rr_distance.py -tgt_s1 /tmp/decMixture_deen_2016_s1 -tgt_s2 /tmp/decMixture_deen_2016_s2 -tgt_ref /tmp/decMixture_deen_2016_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_rank_de.en/data_scores_2016 -cand 5
python experiments/train_model_rr_distance.py -tgt_s1 /tmp/decMixture_deen_2016_s1 -tgt_s2 /tmp/decMixture_deen_2016_s2 -tgt_ref /tmp/decMixture_deen_2016_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_rank_de.en/data_scores_2016 -cand 3
echo '***** decoderState1 *****'
python experiments/train_model_rr_distance.py -tgt_s1 /tmp/decMixture_deen_2016_s1 -tgt_s2 /tmp/decMixture_deen_2016_s2 -tgt_ref /tmp/decMixture_deen_2016_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_rank_de.en/data_scores_2016 -cand 8
python experiments/train_model_rr_distance.py -tgt_s1 /tmp/decMixture_deen_2016_s1 -tgt_s2 /tmp/decMixture_deen_2016_s2 -tgt_ref /tmp/decMixture_deen_2016_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_rank_de.en/data_scores_2016 -cand 10
python experiments/train_model_rr_distance.py -tgt_s1 /tmp/decMixture_deen_2016_s1 -tgt_s2 /tmp/decMixture_deen_2016_s2 -tgt_ref /tmp/decMixture_deen_2016_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_rank_de.en/data_scores_2016 -cand 6
echo '***** decoderState2 *****'
python experiments/train_model_rr_distance.py -tgt_s1 /tmp/decMixture_deen_2016_s1 -tgt_s2 /tmp/decMixture_deen_2016_s2 -tgt_ref /tmp/decMixture_deen_2016_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_rank_de.en/data_scores_2016 -cand 9
python experiments/train_model_rr_distance.py -tgt_s1 /tmp/decMixture_deen_2016_s1 -tgt_s2 /tmp/decMixture_deen_2016_s2 -tgt_ref /tmp/decMixture_deen_2016_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_rank_de.en/data_scores_2016 -cand 11
python experiments/train_model_rr_distance.py -tgt_s1 /tmp/decMixture_deen_2016_s1 -tgt_s2 /tmp/decMixture_deen_2016_s2 -tgt_ref /tmp/decMixture_deen_2016_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_rank_de.en/data_scores_2016 -cand 7
echo '***** decoderCeil1 *****'
python experiments/train_model_rr_distance.py -tgt_s1 /tmp/decMixture_deen_2016_s1 -tgt_s2 /tmp/decMixture_deen_2016_s2 -tgt_ref /tmp/decMixture_deen_2016_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_rank_de.en/data_scores_2016 -cand 14
python experiments/train_model_rr_distance.py -tgt_s1 /tmp/decMixture_deen_2016_s1 -tgt_s2 /tmp/decMixture_deen_2016_s2 -tgt_ref /tmp/decMixture_deen_2016_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_rank_de.en/data_scores_2016 -cand 16
python experiments/train_model_rr_distance.py -tgt_s1 /tmp/decMixture_deen_2016_s1 -tgt_s2 /tmp/decMixture_deen_2016_s2 -tgt_ref /tmp/decMixture_deen_2016_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_rank_de.en/data_scores_2016 -cand 12
echo '***** decoderCeil2 *****'
python experiments/train_model_rr_distance.py -tgt_s1 /tmp/decMixture_deen_2016_s1 -tgt_s2 /tmp/decMixture_deen_2016_s2 -tgt_ref /tmp/decMixture_deen_2016_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_rank_de.en/data_scores_2016 -cand 15
python experiments/train_model_rr_distance.py -tgt_s1 /tmp/decMixture_deen_2016_s1 -tgt_s2 /tmp/decMixture_deen_2016_s2 -tgt_ref /tmp/decMixture_deen_2016_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_rank_de.en/data_scores_2016 -cand 17
python experiments/train_model_rr_distance.py -tgt_s1 /tmp/decMixture_deen_2016_s1 -tgt_s2 /tmp/decMixture_deen_2016_s2 -tgt_ref /tmp/decMixture_deen_2016_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_rank_de.en/data_scores_2016 -cand 13















