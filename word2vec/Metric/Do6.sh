#!/bin/bash
export PYTHONPATH=$PYTHONPATH:../../utils:../../experiments:../..:../../models/
# run get_val_emb-nn

echo '###############################################'
echo '###encoder_hidden mode: sum ###################'
echo '###############################################'
# use the ende to get the en vector
echo '+++++++deen 16+++++++++++'
python scripts/get_valTauLike2.emb-nn.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2016/data_scores -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_source -mode sum -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_ende -type encoder_hidden -test_s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s1 -test_s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s2 -test_ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_ref -test_scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2015/data_scores -test_src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_source  

echo '+++++++deen 15+++++++++++'
python scripts/get_valTauLike2.emb-nn.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2015/data_scores -mode sum -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_ende -type encoder_hidden -test_s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s1 -test_s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s2 -test_ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_ref -test_scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2016/data_scores 




rm /tmp/de*

echo '###############################################'
echo '###encoder_embd mode: sum ###################'
echo '###############################################'
# use the ende to get the en vector
echo '+++++++deen 16+++++++++++'
python scripts/get_valTauLike2.emb-nn.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2016/data_scores -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_source -mode sum -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_ende -type encoder_embd -test_s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s1 -test_s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s2 -test_ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_ref -test_scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2015/data_scores -test_src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_source  

echo '+++++++deen 15+++++++++++'
python scripts/get_valTauLike2.emb-nn.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2015/data_scores -mode sum -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_ende -type encoder_embd -test_s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s1 -test_s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s2 -test_ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_ref -test_scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2016/data_scores 


echo '###############################################'
echo '###encoder_embd mode: max ###################'
echo '###############################################'
# use the ende to get the en vector
echo '+++++++deen 16+++++++++++'
python scripts/get_valTauLike2.emb-nn.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2016/data_scores -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_source -mode max -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_ende -type encoder_embd -test_s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s1 -test_s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s2 -test_ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_ref -test_scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2015/data_scores -test_src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_source  

echo '+++++++deen 15+++++++++++'
python scripts/get_valTauLike2.emb-nn.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2015/data_scores -mode max -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_ende -type encoder_embd -test_s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s1 -test_s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s2 -test_ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_ref -test_scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2016/data_scores 


rm /tmp/de*





echo '###############################################'
echo '###decoder_embd mode: sum ###################'
echo '###############################################'
# use the deen to get the en vector
echo '+++++++deen 16+++++++++++'
python scripts/get_valTauLike2.emb-nn.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2016/data_scores -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_source -mode sum -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_deen -type decoder_embd -test_s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s1 -test_s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s2 -test_ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_ref -test_scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2015/data_scores -test_src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_source  

echo '+++++++deen 15+++++++++++'
python scripts/get_valTauLike2.emb-nn.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2015/data_scores -mode sum -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_deen -type decoder_embd -test_s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s1 -test_s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s2 -test_ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_ref -test_scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2016/data_scores 


echo '###############################################'
echo '###decoder_embd mode: max ###################'
echo '###############################################'
# use the deen to get the en vector
echo '+++++++deen 16+++++++++++'
python scripts/get_valTauLike2.emb-nn.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2016/data_scores -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_source -mode max -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_deen -type decoder_embd -test_s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s1 -test_s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s2 -test_ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_ref -test_scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2015/data_scores -test_src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_source  

echo '+++++++deen 15+++++++++++'
python scripts/get_valTauLike2.emb-nn.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2015/data_scores -mode max -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_deen -type decoder_embd -test_s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s1 -test_s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s2 -test_ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_ref -test_scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2016/data_scores 


rm /tmp/de*













echo '###############################################'
echo '###encoder_hidden mode: sum ###################'
echo '###############################################'
# use the deen to get the en vector
echo '+++++++ende 16+++++++++++'
python scripts/get_valTauLike2.emb-nn.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2016/data_scores -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_source -mode sum -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_deen -type encoder_hidden -test_s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_s1 -test_s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_s2 -test_ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_ref -test_scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2015/data_scores -test_src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_source  

echo '+++++++ende 15+++++++++++'
python scripts/get_valTauLike2.emb-nn.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2015/data_scores -mode sum -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_deen -type encoder_hidden -test_s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_s1 -test_s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_s2 -test_ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_ref -test_scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2016/data_scores 


rm /tmp/en*



echo '###############################################'
echo '###encoder_embd mode: sum ###################'
echo '###############################################'
# use the deen to get the en vector
echo '+++++++ende 16+++++++++++'
python scripts/get_valTauLike2.emb-nn.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2016/data_scores -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_source -mode sum -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_deen -type encoder_embd -test_s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_s1 -test_s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_s2 -test_ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_ref -test_scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2015/data_scores -test_src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_source  

echo '+++++++ende 15+++++++++++'
python scripts/get_valTauLike2.emb-nn.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2015/data_scores -mode sum -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_deen -type encoder_embd -test_s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_s1 -test_s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_s2 -test_ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_ref -test_scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2016/data_scores 


echo '###############################################'
echo '###encoder_embd mode: max ###################'
echo '###############################################'
# use the deen to get the en vector
echo '+++++++ende 16+++++++++++'
python scripts/get_valTauLike2.emb-nn.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2016/data_scores -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_source -mode max -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_deen -type encoder_embd -test_s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_s1 -test_s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_s2 -test_ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_ref -test_scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2015/data_scores -test_src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_source  

echo '+++++++ende 15+++++++++++'
python scripts/get_valTauLike2.emb-nn.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2015/data_scores -mode max -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_deen -type encoder_embd -test_s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_s1 -test_s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_s2 -test_ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_ref -test_scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2016/data_scores 


rm /tmp/en*





echo '###############################################'
echo '###decoder_embd mode: sum ###################'
echo '###############################################'
# use the ende to get the en vector
echo '+++++++ende 16+++++++++++'
python scripts/get_valTauLike2.emb-nn.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2016/data_scores -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_source -mode sum -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_ende -type decoder_embd -test_s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_s1 -test_s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_s2 -test_ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_ref -test_scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2015/data_scores -test_src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_source  

echo '+++++++ende 15+++++++++++'
python scripts/get_valTauLike2.emb-nn.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2015/data_scores -mode sum -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_ende -type decoder_embd -test_s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_s1 -test_s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_s2 -test_ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_ref -test_scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2016/data_scores 


echo '###############################################'
echo '###decoder_embd mode: max ###################'
echo '###############################################'
# use the ende to get the en vector
echo '+++++++ende 16+++++++++++'
python scripts/get_valTauLike2.emb-nn.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2016/data_scores -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_source -mode max -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_ende -type decoder_embd -test_s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_s1 -test_s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_s2 -test_ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_ref -test_scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2015/data_scores -test_src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_source  

echo '+++++++ende 15+++++++++++'
python scripts/get_valTauLike2.emb-nn.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2015/data_scores -mode max -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_ende -type decoder_embd -test_s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_s1 -test_s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_s2 -test_ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_ref -test_scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2016/data_scores 


echo '###############################################'
echo '###decoder_hidden mode: max ###################'
echo '###############################################'
# use the ende to get the en vector
echo '+++++++ende 16+++++++++++'
python scripts/get_valTauLike2.emb-nn.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2016/data_scores -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_source -mode max -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_ende -type decoder_hidden -test_s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_s1 -test_s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_s2 -test_ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_ref -test_scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2015/data_scores -test_src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_source  

echo '+++++++ende 15+++++++++++'
python scripts/get_valTauLike2.emb-nn.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2015/data_scores -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_source -mode max -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_ende -type decoder_hidden -test_s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_s1 -test_s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_s2 -test_ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_ref -test_scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2016/data_scores -test_src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_source  













echo '###############################################'
echo '###decoder_hidden mode: sum ###################'
echo '###############################################'

echo '+++++++deen 16+++++++++++'
python scripts/get_valTauLike2.emb-nn.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2016/data_scores -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_source -mode sum -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_deen -type decoder_hidden -test_s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s1 -test_s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s2 -test_ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_ref -test_scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2015/data_scores -test_src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_source  


echo '+++++++deen 15+++++++++++'
python scripts/get_valTauLike2.emb-nn.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2015/data_scores -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_source -mode sum -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_deen -type decoder_hidden -test_s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s1 -test_s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s2 -test_ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_ref -test_scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2016/data_scores -test_src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_source  

echo '###############################################'
echo '###decoder_hidden mode: max ###################'
echo '###############################################'

echo '+++++++deen 16+++++++++++'
python scripts/get_valTauLike2.emb-nn.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2016/data_scores -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_source -mode max -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_deen -type decoder_hidden -test_s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s1 -test_s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s2 -test_ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_ref -test_scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2015/data_scores -test_src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_source  


echo '+++++++deen 15+++++++++++'
python scripts/get_valTauLike2.emb-nn.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2015/data_scores -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_source -mode max -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_deen -type decoder_hidden -test_s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s1 -test_s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s2 -test_ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_ref -test_scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2016/data_scores -test_src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_source  


