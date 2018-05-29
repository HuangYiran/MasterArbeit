#!/bin/bash
# run get_valTauLike2.emb.py

echo '###########################################'
echo '#### decoder_embd mode: sum ###############'
echo '###########################################'

echo '+++++++deen 16+++++++++++'
python scripts/get_valTauLike2.emb.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2016/data_scores -mode sum -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_deen -type decoder_embd

echo '+++++++deen 15+++++++++++'
python scripts/get_valTauLike2.emb.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2015/data_scores -mode sum -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_deen -type decoder_embd

echo '+++++++deen 14+++++++++++'
python scripts/get_valTauLike2.emb.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2014/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2014/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2014/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2014/data_scores -mode sum -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_deen -type decoder_embd

echo '+++++++deen 13+++++++++++'
python scripts/get_valTauLike2.emb.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2013/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2013/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2013/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2013/data_scores -mode sum -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_deen -type decoder_embd

echo '+++++++ende 16+++++++++++'
python scripts/get_valTauLike2.emb.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2016/data_scores -mode sum -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_ende -type decoder_embd

echo '+++++++ende 15+++++++++++'
python scripts/get_valTauLike2.emb.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2015/data_scores -mode sum -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_ende -type decoder_embd

echo '+++++++ende 14+++++++++++'
python scripts/get_valTauLike2.emb.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2014/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2014/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2014/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2014/data_scores -mode sum -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_ende -type decoder_embd

echo '+++++++ende 13+++++++++++'
python scripts/get_valTauLike2.emb.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2013/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2013/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2013/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2013/data_scores -mode sum -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_ende -type decoder_embd

echo '###########################################'
echo '#### decoder_embd mode: max ###############'
echo '###########################################'

echo '+++++++deen 16+++++++++++'
python scripts/get_valTauLike2.emb.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2016/data_scores -mode max -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_deen -type decoder_embd

echo '+++++++deen 15+++++++++++'
python scripts/get_valTauLike2.emb.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2015/data_scores -mode max -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_deen -type decoder_embd

echo '+++++++deen 14+++++++++++'
python scripts/get_valTauLike2.emb.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2014/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2014/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2014/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2014/data_scores -mode max -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_deen -type decoder_embd

echo '+++++++deen 13+++++++++++'
python scripts/get_valTauLike2.emb.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2013/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2013/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2013/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2013/data_scores -mode max -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_deen -type decoder_embd

echo '+++++++ende 16+++++++++++'
python scripts/get_valTauLike2.emb.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2016/data_scores -mode max -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_ende -type decoder_embd

echo '+++++++ende 15+++++++++++'
python scripts/get_valTauLike2.emb.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2015/data_scores -mode max -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_ende -type decoder_embd

echo '+++++++ende 14+++++++++++'
python scripts/get_valTauLike2.emb.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2014/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2014/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2014/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2014/data_scores -mode max -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_ende -type decoder_embd

echo '+++++++ende 13+++++++++++'
python scripts/get_valTauLike2.emb.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2013/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2013/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2013/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2013/data_scores -mode max -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_ende -type decoder_embd


echo '###########################################'
echo '###encoder_embd mode: sum #################'
echo '###########################################'
# use the ende to get the en vector
echo '+++++++deen 16+++++++++++'
python scripts/get_valTauLike2.emb.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2016/data_scores -mode sum -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_ende -type encoder_embd

echo '+++++++deen 15+++++++++++'
python scripts/get_valTauLike2.emb.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2015/data_scores -mode sum -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_ende -type encoder_embd

echo '+++++++deen 14+++++++++++'
python scripts/get_valTauLike2.emb.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2014/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2014/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2014/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2014/data_scores -mode sum -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_ende -type encoder_embd

echo '+++++++deen 13+++++++++++'
python scripts/get_valTauLike2.emb.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2013/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2013/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2013/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2013/data_scores -mode sum -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_ende -type encoder_embd

echo '+++++++ende 16+++++++++++'
python scripts/get_valTauLike2.emb.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2016/data_scores -mode sum -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_deen -type encoder_embd

echo '+++++++ende 15+++++++++++'
python scripts/get_valTauLike2.emb.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2015/data_scores -mode sum -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_deen -type encoder_embd

echo '+++++++ende 14+++++++++++'
python scripts/get_valTauLike2.emb.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2014/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2014/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2014/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2014/data_scores -mode sum -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_deen -type encoder_embd

echo '+++++++ende 13+++++++++++'
python scripts/get_valTauLike2.emb.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2013/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2013/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2013/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2013/data_scores -mode sum -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_deen -type encoder_embd

echo '###########################################'
echo '###encoder_embd mode: max #################'
echo '###########################################'
# use the ende to get the en vector
echo '+++++++deen 16+++++++++++'
python scripts/get_valTauLike2.emb.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2016/data_scores -mode max -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_ende -type encoder_embd

echo '+++++++deen 15+++++++++++'
python scripts/get_valTauLike2.emb.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2015/data_scores -mode max -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_ende -type encoder_embd

echo '+++++++deen 14+++++++++++'
python scripts/get_valTauLike2.emb.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2014/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2014/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2014/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2014/data_scores -mode max -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_ende -type encoder_embd

echo '+++++++deen 13+++++++++++'
python scripts/get_valTauLike2.emb.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2013/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2013/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2013/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2013/data_scores -mode max -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_ende -type encoder_embd

echo '+++++++ende 16+++++++++++'
python scripts/get_valTauLike2.emb.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2016/data_scores -mode max -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_deen -type encoder_embd

echo '+++++++ende 15+++++++++++'
python scripts/get_valTauLike2.emb.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2015/data_scores -mode max -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_deen -type encoder_embd

echo '+++++++ende 14+++++++++++'
python scripts/get_valTauLike2.emb.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2014/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2014/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2014/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2014/data_scores -mode max -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_deen -type encoder_embd

echo '+++++++ende 13+++++++++++'
python scripts/get_valTauLike2.emb.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2013/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2013/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2013/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2013/data_scores -mode max -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_deen -type encoder_embd


echo '###############################################'
echo '###encoder_hidden mode: sum ###################'
echo '###############################################'
# use the ende to get the en vector
echo '+++++++deen 16+++++++++++'
python scripts/get_valTauLike2.emb.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2016/data_scores -mode sum -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_ende -type encoder_hidden

echo '+++++++deen 15+++++++++++'
python scripts/get_valTauLike2.emb.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2015/data_scores -mode sum -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_ende -type encoder_hidden

echo '+++++++deen 14+++++++++++'
python scripts/get_valTauLike2.emb.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2014/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2014/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2014/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2014/data_scores -mode sum -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_ende -type encoder_hidden

echo '+++++++deen 13+++++++++++'
python scripts/get_valTauLike2.emb.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2013/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2013/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2013/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2013/data_scores -mode sum -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_ende -type encoder_hidden

echo '+++++++ende 16+++++++++++'
python scripts/get_valTauLike2.emb.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2016/data_scores -mode sum -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_deen -type encoder_hidden

echo '+++++++ende 15+++++++++++'
python scripts/get_valTauLike2.emb.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2015/data_scores -mode sum -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_deen -type encoder_hidden

echo '+++++++ende 14+++++++++++'
python scripts/get_valTauLike2.emb.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2014/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2014/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2014/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2014/data_scores -mode sum -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_deen -type encoder_hidden

echo '+++++++ende 13+++++++++++'
python scripts/get_valTauLike2.emb.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2013/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2013/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2013/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2013/data_scores -mode sum -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_deen -type encoder_hidden

echo '#############################################'
echo '#### decoder_hidden mode: sum ###############'
echo '#############################################'

echo '+++++++deen 16+++++++++++'
python scripts/get_valTauLike2.emb.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2016/data_scores -mode sum -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_deen -type decoder_hidden -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_source 

echo '+++++++deen 15+++++++++++'
python scripts/get_valTauLike2.emb.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2015/data_scores -mode sum -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_deen -type decoder_hidden -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_source 

echo '+++++++deen 14+++++++++++'
python scripts/get_valTauLike2.emb.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2014/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2014/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2014/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2014/data_scores -mode sum -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_deen -type decoder_hidden -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2014/data_source 

echo '+++++++deen 13+++++++++++'
python scripts/get_valTauLike2.emb.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2013/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2013/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2013/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2013/data_scores -mode sum -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_deen -type decoder_hidden -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2013/data_source 

echo '+++++++ende 16+++++++++++'
python scripts/get_valTauLike2.emb.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2016/data_scores -mode sum -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_ende -type decoder_hidden -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_source 

echo '+++++++ende 15+++++++++++'
python scripts/get_valTauLike2.emb.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2015/data_scores -mode sum -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_ende -type decoder_hidden -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_source 

echo '+++++++ende 14+++++++++++'
python scripts/get_valTauLike2.emb.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2014/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2014/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2014/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2014/data_scores -mode sum -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_ende -type decoder_hidden -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2014/data_source 

echo '+++++++ende 13+++++++++++'
python scripts/get_valTauLike2.emb.py -s1 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2013/data_s1 -s2 /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2013/data_s2 -ref /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2013/data_ref -scores /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2013/data_scores -mode sum -model /Users/ihuangyiran/Documents/Workplace_Python/data/mt_model/model_ende -type decoder_hidden -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2013/data_source 


