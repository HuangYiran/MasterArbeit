#!/bin/bash

#python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type encoder -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_source -output ../data/MasterArbeit/plan_c_rank_de.en/encoder_embd/data_ref
#python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type decoder -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_ref -output ../data/MasterArbeit/plan_c_rank_de.en/decoder_embd/data_ref
#python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type encoder_hidden -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_source -output ../data/MasterArbeit/plan_c_rank_de.en/encoder_hidden/data_ref
#python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type decoder_hidden -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_source -tgt /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_ref -output ../data/MasterArbeit/plan_c_rank_de.en/decoder_hidden/data_test

echo '############################'
echo '####decoder_embds ##########'
echo '############################'
# be careful that, use deen model to get the en embedding 
echo '++++++++++deen 16+++++++++++'
python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type decoder_embd -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_ref -output ../data/MasterArbeit/plan_c_rank_de.en/decoder_embd/16/data_ref
python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type decoder_embd -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s1 -output ../data/MasterArbeit/plan_c_rank_de.en/decoder_embd/16/data_s1
python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type decoder_embd -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s2 -output ../data/MasterArbeit/plan_c_rank_de.en/decoder_embd/16/data_s2

echo '++++++++++deen 15++++++++++'
python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type decoder_embd -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_ref -output ../data/MasterArbeit/plan_c_rank_de.en/decoder_embd/15/data_ref
python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type decoder_embd -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s1 -output ../data/MasterArbeit/plan_c_rank_de.en/decoder_embd/15/data_s1
python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type decoder_embd -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s2 -output ../data/MasterArbeit/plan_c_rank_de.en/decoder_embd/15/data_s2

echo '++++++++++deen 14++++++++++'
python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type decoder_embd -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2014/data_ref -output ../data/MasterArbeit/plan_c_rank_de.en/decoder_embd/14/data_ref
python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type decoder_embd -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2014/data_s1 -output ../data/MasterArbeit/plan_c_rank_de.en/decoder_embd/14/data_s1
python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type decoder_embd -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2014/data_s2 -output ../data/MasterArbeit/plan_c_rank_de.en/decoder_embd/14/data_s2

echo '++++++++++deen 13++++++++++'
python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type decoder_embd -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2013/data_ref -output ../data/MasterArbeit/plan_c_rank_de.en/decoder_embd/13/data_ref
python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type decoder_embd -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2013/data_s1 -output ../data/MasterArbeit/plan_c_rank_de.en/decoder_embd/13/data_s1
python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type decoder_embd -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2013/data_s2 -output ../data/MasterArbeit/plan_c_rank_de.en/decoder_embd/13/data_s2

echo '++++++++++ende 16++++++++++'
python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_ende -type decoder_embd -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_ref -output ../data/MasterArbeit/plan_c_rank_en.de/decoder_embd/16/data_ref
python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_ende -type decoder_embd -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_s1 -output ../data/MasterArbeit/plan_c_rank_en.de/decoder_embd/16/data_s1
python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_ende -type decoder_embd -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_s2 -output ../data/MasterArbeit/plan_c_rank_en.de/decoder_embd/16/data_s2

echo '++++++++++ende 15++++++++++'
python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_ende -type decoder_embd -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_ref -output ../data/MasterArbeit/plan_c_rank_en.de/decoder_embd/15/data_ref
python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_ende -type decoder_embd -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_s1 -output ../data/MasterArbeit/plan_c_rank_en.de/decoder_embd/15/data_s1
python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_ende -type decoder_embd -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_s2 -output ../data/MasterArbeit/plan_c_rank_en.de/decoder_embd/15/data_s2

echo '++++++++++ende 14++++++++++'
python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_ende -type decoder_embd -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2014/data_ref -output ../data/MasterArbeit/plan_c_rank_en.de/decoder_embd/14/data_ref
python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_ende -type decoder_embd -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2014/data_s1 -output ../data/MasterArbeit/plan_c_rank_en.de/decoder_embd/14/data_s1
python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_ende -type decoder_embd -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2014/data_s2 -output ../data/MasterArbeit/plan_c_rank_en.de/decoder_embd/14/data_s2

echo '++++++++++ende 13++++++++++'
python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_ende -type decoder_embd -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2013/data_ref -output ../data/MasterArbeit/plan_c_rank_en.de/decoder_embd/13/data_ref
python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_ende -type decoder_embd -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2013/data_s1 -output ../data/MasterArbeit/plan_c_rank_en.de/decoder_embd/13/data_s1
python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_ende -type decoder_embd -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2013/data_s2 -output ../data/MasterArbeit/plan_c_rank_en.de/decoder_embd/13/data_s2

echo '############################'
echo '####encoder_hidden #########'
echo '############################'
# different from decoder_embd, we use ende model to get the en data
echo '++++++++++deen 16+++++++++++'
python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_ende -type encoder_hidden -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_ref -output ../data/MasterArbeit/plan_c_rank_de.en/encoder_hidden/16/data_ref
python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_ende -type encoder_hidden -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s1 -output ../data/MasterArbeit/plan_c_rank_de.en/encoder_hidden/16/data_s1
python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_ende -type encoder_hidden -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2016/data_s2 -output ../data/MasterArbeit/plan_c_rank_de.en/encoder_hidden/16/data_s2

echo '++++++++++deen 15++++++++++'
python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_ende -type encoder_hidden -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_ref -output ../data/MasterArbeit/plan_c_rank_de.en/encoder_hidden/15/data_ref
python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_ende -type encoder_hidden -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s1 -output ../data/MasterArbeit/plan_c_rank_de.en/encoder_hidden/15/data_s1
python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_ende -type encoder_hidden -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2015/data_s2 -output ../data/MasterArbeit/plan_c_rank_de.en/encoder_hidden/15/data_s2

echo '++++++++++deen 14++++++++++'
python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_ende -type encoder_hidden -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2014/data_ref -output ../data/MasterArbeit/plan_c_rank_de.en/encoder_hidden/14/data_ref
python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_ende -type encoder_hidden -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2014/data_s1 -output ../data/MasterArbeit/plan_c_rank_de.en/encoder_hidden/14/data_s1
python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_ende -type encoder_hidden -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2014/data_s2 -output ../data/MasterArbeit/plan_c_rank_de.en/encoder_hidden/14/data_s2

echo '++++++++++deen 13++++++++++'
python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_ende -type encoder_hidden -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2013/data_ref -output ../data/MasterArbeit/plan_c_rank_de.en/encoder_hidden/13/data_ref
python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_ende -type encoder_hidden -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2013/data_s1 -output ../data/MasterArbeit/plan_c_rank_de.en/encoder_hidden/13/data_s1
python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_ende -type encoder_hidden -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/prepro_data_2013/data_s2 -output ../data/MasterArbeit/plan_c_rank_de.en/encoder_hidden/13/data_s2

echo '++++++++++ende 16++++++++++'
python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type encoder_hidden -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_ref -output ../data/MasterArbeit/plan_c_rank_en.de/encoder_hidden/16/data_ref
python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type encoder_hidden -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_s1 -output ../data/MasterArbeit/plan_c_rank_en.de/encoder_hidden/16/data_s1
python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type encoder_hidden -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2016/data_s2 -output ../data/MasterArbeit/plan_c_rank_en.de/encoder_hidden/16/data_s2

echo '++++++++++ende 15++++++++++'
python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type encoder_hidden -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_ref -output ../data/MasterArbeit/plan_c_rank_en.de/encoder_hidden/15/data_ref
python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type encoder_hidden -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_s1 -output ../data/MasterArbeit/plan_c_rank_en.de/encoder_hidden/15/data_s1
python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type encoder_hidden -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2015/data_s2 -output ../data/MasterArbeit/plan_c_rank_en.de/encoder_hidden/15/data_s2

echo '++++++++++ende 14++++++++++'
python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type encoder_hidden -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2014/data_ref -output ../data/MasterArbeit/plan_c_rank_en.de/encoder_hidden/14/data_ref
python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type encoder_hidden -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2014/data_s1 -output ../data/MasterArbeit/plan_c_rank_en.de/encoder_hidden/14/data_s1
python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type encoder_hidden -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2014/data_s2 -output ../data/MasterArbeit/plan_c_rank_en.de/encoder_hidden/14/data_s2

echo '++++++++++ende 13++++++++++'
python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type encoder_hidden -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2013/data_ref -output ../data/MasterArbeit/plan_c_rank_en.de/encoder_hidden/13/data_ref
python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type encoder_hidden -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2013/data_s1 -output ../data/MasterArbeit/plan_c_rank_en.de/encoder_hidden/13/data_s1
python ./get_vector_from_sentence/get_embeddings.py -model ../data/mt_model/model_deen -type encoder_hidden -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_en.de/prepro_data_2013/data_s2 -output ../data/MasterArbeit/plan_c_rank_en.de/encoder_hidden/13/data_s2


