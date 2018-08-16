echo '################################'
echo '##### decoder_mixture ##########'
echo '################################'
#echo 'direct assessment'
#echo 'de-en'
#python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type decoder_mixture -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/prepro_extracted_data_2015/preproplan_c_de_sys_2015.bpe.noUndo.de -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/prepro_extracted_data_2015/preproplan_c_de_sys_2015.bpe.noUndo.en -output /tmp/decMixture_da_2015_sys
#python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type decoder_mixture -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/prepro_extracted_data_2015/preproplan_c_de_sys_2015.bpe.noUndo.de -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/prepro_extracted_data_2015/preproplan_c_de_ref_2015.bpe.noUndo.en -output /tmp/decMixture_da_2015_ref


#python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type decoder_mixture -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/prepro_extracted_data_2017/preproplan_c_de_sys_2017.bpe.noUndo.de -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/prepro_extracted_data_2017/preproplan_c_de_sys_2017.bpe.noUndo.en -output /tmp/decMixture_da_2017_sys
#python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type decoder_mixture -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/prepro_extracted_data_2017/preproplan_c_de_sys_2017.bpe.noUndo.de -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/prepro_extracted_data_2017/preproplan_c_de_ref_2017.bpe.noUndo.en -output /tmp/decMixture_da_2017_ref
#
#python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type decoder_mixture -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/prepro_extracted_data_2016/preproplan_c_de_sys_2016.bpe.noUndo.de -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/prepro_extracted_data_2016/preproplan_c_de_sys_2016.bpe.noUndo.en -output /tmp/decMixture_da_2016_sys
#python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type decoder_mixture -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/prepro_extracted_data_2016/preproplan_c_de_sys_2016.bpe.noUndo.de -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/prepro_extracted_data_2016/preproplan_c_de_ref_2016.bpe.noUndo.en -output /tmp/decMixture_da_2016_ref




echo 'relative ranking'
echo 'de-en'
#python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type decoder_mixture -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_ref -output /tmp/decMixture_2015_deen_ref
#python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type decoder_mixture -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s1 -output /tmp/decMixture_2015_deen_s1
#python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type decoder_mixture -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s2 -output /tmp/decMixture_2015_deen_s2
#
python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type decoder_mixture -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2014/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2014/data_ref -output /tmp/decMixture_2014_deen_ref
python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type decoder_mixture -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2014/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2014/data_s1 -output /tmp/decMixture_2014_deen_s1
python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type decoder_mixture -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2014/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2014/data_s2 -output /tmp/decMixture_2014_deen_s2


echo 'en-de'
python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_ende -type decoder_mixture -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_ref -output /tmp/decMixture_2016_ende_ref
python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_ende -type decoder_mixture -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_s1 -output /tmp/decMixture_2016_ende_s1
python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_ende -type decoder_mixture -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_s2 -output /tmp/decMixture_2016_ende_s2


echo '###########################'
echo '##### decoder_wr ##########'
echo '###########################'
#python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type decoder_wr -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_ref -output /tmp/decWR_2016_ref
#python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type decoder_wr -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s1 -output /tmp/decWR_2016_s1
#python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type decoder_wr -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s2 -output /tmp/decWR_2016_s2

#python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type decoder_wr -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_ref -output /tmp/decWR_2015_ref
#python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type decoder_wr -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s1 -output /tmp/decWR_2015_s1
#python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type decoder_wr -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s2 -output /tmp/decWR_2015_s2



#echo '###########################'
#echo '#####decoder_states########'
#echo '###########################'

#python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type decoder_states -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_ref -output /tmp/decStates_2016_ref
#python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type decoder_states -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s1 -output /tmp/decStates_2016_s1
#python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type decoder_states -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s2 -output /tmp/decStates_2016_s2

#python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type decoder_states -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_ref -output /tmp/decStates_2015_ref
#python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type decoder_states -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s1 -output /tmp/decStates_2015_s1
#python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type decoder_states -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s2 -output /tmp/decStates_2015_s2

###########################
#####decoder_ceil##########
###########################
#python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type decoder_ceil -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_ref -output /tmp/decCeil_2016_ref
#python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type decoder_ceil -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s1 -output /tmp/decCeil_2016_s1
#python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type decoder_ceil -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s2 -output /tmp/decCeil_2016_s2
#
#python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type decoder_ceil -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_ref -output /tmp/decCeil_2015_ref
#python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type decoder_ceil -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s1 -output /tmp/decCeil_2015_s1
#python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type decoder_ceil -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s2 -output /tmp/decCeil_2015_s2

##############################
##### decoder mean ###########
##############################
#python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type decoder_hidden_mean -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_ref -output /tmp/decMean_2016_ref
#python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type decoder_hidden_mean -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s1 -output /tmp/decMean_2016_s1
#python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type decoder_hidden_mean -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s2 -output /tmp/decMean_2016_s2
#
#python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type decoder_hidden_mean -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_ref -output /tmp/decMean_2015_ref
#python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type decoder_hidden_mean -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s1 -output /tmp/decMean_2015_s1
#python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type decoder_hidden_mean -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s2 -output /tmp/decMean_2015_s2

#echo '##############################'
#echo '##### decoder sum  ##########'
#echo '##############################'
#python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type decoder_hidden_sum -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_ref -output /tmp/decSum_2016_ref
#python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type decoder_hidden_sum -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s1 -output /tmp/decSum_2016_s1
#python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type decoder_hidden_sum -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s2 -output /tmp/decSum_2016_s2

#python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type decoder_hidden_sum -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_ref -output /tmp/decSum_2015_ref
#python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type decoder_hidden_sum -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s1 -output /tmp/decSum_2015_s1
#python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type decoder_hidden_sum -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s2 -output /tmp/decSum_2015_s2


