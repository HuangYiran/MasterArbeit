# get all the embeddings, that are used in the experiment including: w2v, encoder, decoder

# w2v
echo '###########################'
echo '######### w2v slim ########'
echo '###########################'
echo '***** max ******'
echo '---- 2015 ----'
python word2vec/process_w2v_for_file.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2015/data_reference.de-en -out /tmp/w2v_deen_2015_ref_slim -model word2vec/models/english/GoogleNews-vectors-negative300-SLIM.bin --bin -agg max
python word2vec/process_w2v_for_file.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2015/data_source.de-en  -out /tmp/w2v_deen_2015_src_slim -model word2vec/models/english/GoogleNews-vectors-negative300-SLIM.bin --bin -agg max
python word2vec/process_w2v_for_file.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2015/data_sys_out.de-en -out /tmp/w2v_deen_2015_sys_slim -model word2vec/models/english/GoogleNews-vectors-negative300-SLIM.bin --bin -agg max
echo '---- 2016 ----'
python word2vec/process_w2v_for_file.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2016/DAseg.newstest2016.reference.de-en -out /tmp/w2v_deen_2016_ref_slim -model word2vec/models/english/GoogleNews-vectors-negative300-SLIM.bin --bin -agg max
python word2vec/process_w2v_for_file.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2016/DAseg.newstest2016.source.de-en -out /tmp/w2v_deen_2016_src_slim -model word2vec/models/english/GoogleNews-vectors-negative300-SLIM.bin --bin -agg max
python word2vec/process_w2v_for_file.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2016/DAseg.newstest2016.mt-system.de-en -out /tmp/w2v_deen_2016_sys_slim -model word2vec/models/english/GoogleNews-vectors-negative300-SLIM.bin --bin -agg max
echo '---- 2017 ----'
python word2vec/process_w2v_for_file.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2017/data_ref -out /tmp/w2v_deen_2017_ref_slim -model word2vec/models/english/GoogleNews-vectors-negative300-SLIM.bin --bin -agg max
python word2vec/process_w2v_for_file.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2017/data_source -out /tmp/w2v_deen_2017_src_slim -model word2vec/models/english/GoogleNews-vectors-negative300-SLIM.bin --bin -agg max
python word2vec/process_w2v_for_file.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2017/data_sys_out -out /tmp/w2v_deen_2017_sys_slim -model word2vec/models/english/GoogleNews-vectors-negative300-SLIM.bin --bin -agg max

echo '***** sum ******'
echo '---- 2015 ----'
python word2vec/process_w2v_for_file.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2015/data_reference.de-en -out /tmp/w2v_deen_2015_ref_slim -model word2vec/models/english/GoogleNews-vectors-negative300-SLIM.bin --bin -agg sum
python word2vec/process_w2v_for_file.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2015/data_source.de-en  -out /tmp/w2v_deen_2015_src_slim -model word2vec/models/english/GoogleNews-vectors-negative300-SLIM.bin --bin -agg sum
python word2vec/process_w2v_for_file.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2015/data_sys_out.de-en -out /tmp/w2v_deen_2015_sys_slim -model word2vec/models/english/GoogleNews-vectors-negative300-SLIM.bin --bin -agg sum
echo '---- 2016 ----'
python word2vec/process_w2v_for_file.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2016/DAseg.newstest2016.reference.de-en -out /tmp/w2v_deen_2016_ref_slim -model word2vec/models/english/GoogleNews-vectors-negative300-SLIM.bin --bin -agg sum
python word2vec/process_w2v_for_file.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2016/DAseg.newstest2016.source.de-en -out /tmp/w2v_deen_2016_src_slim -model word2vec/models/english/GoogleNews-vectors-negative300-SLIM.bin --bin -agg sum
python word2vec/process_w2v_for_file.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2016/DAseg.newstest2016.mt-system.de-en -out /tmp/w2v_deen_2016_sys_slim -model word2vec/models/english/GoogleNews-vectors-negative300-SLIM.bin --bin -agg sum
echo '---- 2017 ----'
python word2vec/process_w2v_for_file.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2017/data_ref -out /tmp/w2v_deen_2017_ref_slim -model word2vec/models/english/GoogleNews-vectors-negative300-SLIM.bin --bin -agg sum
python word2vec/process_w2v_for_file.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2017/data_source -out /tmp/w2v_deen_2017_src_slim -model word2vec/models/english/GoogleNews-vectors-negative300-SLIM.bin --bin -agg sum
python word2vec/process_w2v_for_file.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2017/data_sys_out -out /tmp/w2v_deen_2017_sys_slim -model word2vec/models/english/GoogleNews-vectors-negative300-SLIM.bin --bin -agg sum
echo '***** mean ******'
echo '---- 2015 ----'
python word2vec/process_w2v_for_file.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2015/data_reference.de-en -out /tmp/w2v_deen_2015_ref_slim -model word2vec/models/english/GoogleNews-vectors-negative300-SLIM.bin --bin -agg mean
python word2vec/process_w2v_for_file.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2015/data_source.de-en  -out /tmp/w2v_deen_2015_src_slim -model word2vec/models/english/GoogleNews-vectors-negative300-SLIM.bin --bin -agg mean
python word2vec/process_w2v_for_file.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2015/data_sys_out.de-en -out /tmp/w2v_deen_2015_sys_slim -model word2vec/models/english/GoogleNews-vectors-negative300-SLIM.bin --bin -agg mean
echo '---- 2016 ----'
python word2vec/process_w2v_for_file.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2016/DAseg.newstest2016.reference.de-en -out /tmp/w2v_deen_2016_ref_slim -model word2vec/models/english/GoogleNews-vectors-negative300-SLIM.bin --bin -agg mean
python word2vec/process_w2v_for_file.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2016/DAseg.newstest2016.source.de-en -out /tmp/w2v_deen_2016_src_slim -model word2vec/models/english/GoogleNews-vectors-negative300-SLIM.bin --bin -agg mean
python word2vec/process_w2v_for_file.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2016/DAseg.newstest2016.mt-system.de-en -out /tmp/w2v_deen_2016_sys_slim -model word2vec/models/english/GoogleNews-vectors-negative300-SLIM.bin --bin -agg mean
echo '---- 2017 ----'
python word2vec/process_w2v_for_file.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2017/data_ref -out /tmp/w2v_deen_2017_ref_slim -model word2vec/models/english/GoogleNews-vectors-negative300-SLIM.bin --bin -agg mean
python word2vec/process_w2v_for_file.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2017/data_source -out /tmp/w2v_deen_2017_src_slim -model word2vec/models/english/GoogleNews-vectors-negative300-SLIM.bin --bin -agg mean
python word2vec/process_w2v_for_file.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2017/data_sys_out -out /tmp/w2v_deen_2017_sys_slim -model word2vec/models/english/GoogleNews-vectors-negative300-SLIM.bin --bin -agg mean


echo '###########################'
echo '########### w2v  ##########'
echo '###########################'
echo '***** max ******'
echo '---- 2015 ----'
python word2vec/process_w2v_for_file.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2015/data_reference.de-en -out /tmp/w2v_deen_2015_ref -model word2vec/models/english/GoogleNews-vectors-negative300.bin --bin -agg max
python word2vec/process_w2v_for_file.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2015/data_source.de-en  -out /tmp/w2v_deen_2015_src -model word2vec/models/english/GoogleNews-vectors-negative300.bin --bin -agg max
python word2vec/process_w2v_for_file.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2015/data_sys_out.de-en -out /tmp/w2v_deen_2015_sys -model word2vec/models/english/GoogleNews-vectors-negative300.bin --bin -agg max
echo '---- 2016 ----'
python word2vec/process_w2v_for_file.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2016/DAseg.newstest2016.reference.de-en -out /tmp/w2v_deen_2016_ref -model word2vec/models/english/GoogleNews-vectors-negative300.bin --bin -agg max
python word2vec/process_w2v_for_file.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2016/DAseg.newstest2016.source.de-en -out /tmp/w2v_deen_2016_src -model word2vec/models/english/GoogleNews-vectors-negative300.bin --bin -agg max
python word2vec/process_w2v_for_file.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2016/DAseg.newstest2016.mt-system.de-en -out /tmp/w2v_deen_2016_sys -model word2vec/models/english/GoogleNews-vectors-negative300.bin --bin -agg max
echo '---- 2017 ----'
python word2vec/process_w2v_for_file.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2017/data_ref -out /tmp/w2v_deen_2017_ref -model word2vec/models/english/GoogleNews-vectors-negative300.bin --bin -agg max
python word2vec/process_w2v_for_file.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2017/data_source -out /tmp/w2v_deen_2017_src -model word2vec/models/english/GoogleNews-vectors-negative300.bin --bin -agg max
python word2vec/process_w2v_for_file.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2017/data_sys_out -out /tmp/w2v_deen_2017_sys -model word2vec/models/english/GoogleNews-vectors-negative300.bin --bin -agg max

echo '***** sum ******'
echo '---- 2015 ----'
python word2vec/process_w2v_for_file.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2015/data_reference.de-en -out /tmp/w2v_deen_2015_ref -model word2vec/models/english/GoogleNews-vectors-negative300.bin --bin -agg sum
python word2vec/process_w2v_for_file.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2015/data_source.de-en  -out /tmp/w2v_deen_2015_src -model word2vec/models/english/GoogleNews-vectors-negative300.bin --bin -agg sum
python word2vec/process_w2v_for_file.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2015/data_sys_out.de-en -out /tmp/w2v_deen_2015_sys -model word2vec/models/english/GoogleNews-vectors-negative300.bin --bin -agg sum
echo '---- 2016 ----'
python word2vec/process_w2v_for_file.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2016/DAseg.newstest2016.reference.de-en -out /tmp/w2v_deen_2016_ref -model word2vec/models/english/GoogleNews-vectors-negative300.bin --bin -agg sum
python word2vec/process_w2v_for_file.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2016/DAseg.newstest2016.source.de-en -out /tmp/w2v_deen_2016_src -model word2vec/models/english/GoogleNews-vectors-negative300.bin --bin -agg sum
python word2vec/process_w2v_for_file.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2016/DAseg.newstest2016.mt-system.de-en -out /tmp/w2v_deen_2016_sys -model word2vec/models/english/GoogleNews-vectors-negative300.bin --bin -agg sum
echo '---- 2017 ----'
python word2vec/process_w2v_for_file.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2017/data_ref -out /tmp/w2v_deen_2017_ref -model word2vec/models/english/GoogleNews-vectors-negative300.bin --bin -agg sum
python word2vec/process_w2v_for_file.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2017/data_source -out /tmp/w2v_deen_2017_src -model word2vec/models/english/GoogleNews-vectors-negative300.bin --bin -agg sum
python word2vec/process_w2v_for_file.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2017/data_sys_out -out /tmp/w2v_deen_2017_sys -model word2vec/models/english/GoogleNews-vectors-negative300.bin --bin -agg sum
echo '***** mean ******'
echo '---- 2015 ----'
python word2vec/process_w2v_for_file.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2015/data_reference.de-en -out /tmp/w2v_deen_2015_ref -model word2vec/models/english/GoogleNews-vectors-negative300.bin --bin -agg mean
python word2vec/process_w2v_for_file.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2015/data_source.de-en  -out /tmp/w2v_deen_2015_src -model word2vec/models/english/GoogleNews-vectors-negative300.bin --bin -agg mean
python word2vec/process_w2v_for_file.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2015/data_sys_out.de-en -out /tmp/w2v_deen_2015_sys -model word2vec/models/english/GoogleNews-vectors-negative300.bin --bin -agg mean
echo '---- 2016 ----'
python word2vec/process_w2v_for_file.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2016/DAseg.newstest2016.reference.de-en -out /tmp/w2v_deen_2016_ref -model word2vec/models/english/GoogleNews-vectors-negative300.bin --bin -agg mean
python word2vec/process_w2v_for_file.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2016/DAseg.newstest2016.source.de-en -out /tmp/w2v_deen_2016_src -model word2vec/models/english/GoogleNews-vectors-negative300.bin --bin -agg mean
python word2vec/process_w2v_for_file.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2016/DAseg.newstest2016.mt-system.de-en -out /tmp/w2v_deen_2016_sys -model word2vec/models/english/GoogleNews-vectors-negative300.bin --bin -agg mean
echo '---- 2017 ----'
python word2vec/process_w2v_for_file.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2017/data_ref -out /tmp/w2v_deen_2017_ref -model word2vec/models/english/GoogleNews-vectors-negative300.bin --bin -agg mean
python word2vec/process_w2v_for_file.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2017/data_source -out /tmp/w2v_deen_2017_src -model word2vec/models/english/GoogleNews-vectors-negative300.bin --bin -agg mean
python word2vec/process_w2v_for_file.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2017/data_sys_out -out /tmp/w2v_deen_2017_sys -model word2vec/models/english/GoogleNews-vectors-negative300.bin --bin -agg mean

echo '###########################'
echo '#### encoder embedding ####'
echo '###########################'
echo '***** mixture-3  ******'
echo '---- 2015 ----'
python ./get_vector_from_sentence/get_embeddings.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2015/data_reference.de-en -out /tmp/encoderEmb_deen_2015_mix_ref -model ../data/mt_model/model_ende -type encoder_mixture
python ./get_vector_from_sentence/get_embeddings.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2015/data_source.de-en  -out /tmp/encoderEmb_deen_2015_mix_src -model ../data/mt_model/model_ende -type encoder_mixture
python ./get_vector_from_sentence/get_embeddings.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2015/data_sys_out.de-en -out /tmp/encoderEmb_deen_2015_mix_sys -model ../data/mt_model/model_ende -type encoder_mixture
echo '---- 2016 ----'
python ./get_vector_from_sentence/get_embeddings.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2016/DAseg.newstest2016.reference.de-en -out /tmp/encoderEmb_deen_2016_mix_ref -model ../data/mt_model/model_ende -type encoder_mixture
python ./get_vector_from_sentence/get_embeddings.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2016/DAseg.newstest2016.source.de-en -out /tmp/encoderEmb_deen_2016_mix_src -model ../data/mt_model/model_ende -type encoder_mixture
python ./get_vector_from_sentence/get_embeddings.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2016/DAseg.newstest2016.mt-system.de-en -out /tmp/encoderEmb_deen_2016_mix_sys -model ../data/mt_model/model_ende -type encoder_mixture
echo '---- 2017 ----'
python ./get_vector_from_sentence/get_embeddings.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2017/data_ref -out /tmp/encoderEmb_deen_2017_mix_ref -model ../data/mt_model/model_ende -type encoder_mixture
python ./get_vector_from_sentence/get_embeddings.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2017/data_source -out /tmp/encoderEmb_deen_2017_mix_src -model ../data/mt_model/model_ende -type encoder_mixture
python ./get_vector_from_sentence/get_embeddings.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2017/data_sys_out -out /tmp/encoderEmb_deen_2017_mix_sys -model ../data/mt_model/model_ende -type encoder_mixture

echo '###########################'
echo '####### encoder output ####'
echo '###########################'

echo '***** encoder_output  ******'
echo '---- 2015 ----'
python ./get_vector_from_sentence/get_embeddings.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2015/data_reference.de-en -out /tmp/encoderEmb_deen_2015_mix_ref -model ../data/mt_model/model_ende -type encoder_hidden
python ./get_vector_from_sentence/get_embeddings.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2015/data_source.de-en  -out /tmp/encoderEmb_deen_2015_mix_src -model ../data/mt_model/model_ende -type encoder_hidden
python ./get_vector_from_sentence/get_embeddings.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2015/data_sys_out.de-en -out /tmp/encoderEmb_deen_2015_mix_sys -model ../data/mt_model/model_ende -type encoder_hidden
echo '---- 2016 ----'
python ./get_vector_from_sentence/get_embeddings.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2016/DAseg.newstest2016.reference.de-en -out /tmp/encoderEmb_deen_2016_mix_ref -model ../data/mt_model/model_ende -type encoder_hidden
python ./get_vector_from_sentence/get_embeddings.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2016/DAseg.newstest2016.source.de-en -out /tmp/encoderEmb_deen_2016_mix_src -model ../data/mt_model/model_ende -type encoder_hidden
python ./get_vector_from_sentence/get_embeddings.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2016/DAseg.newstest2016.mt-system.de-en -out /tmp/encoderEmb_deen_2016_mix_sys -model ../data/mt_model/model_ende -type encoder_hidden
echo '---- 2017 ----'
python ./get_vector_from_sentence/get_embeddings.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2017/data_ref -out /tmp/encoderEmb_deen_2017_mix_ref -model ../data/mt_model/model_ende -type encoder_hidden
python ./get_vector_from_sentence/get_embeddings.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2017/data_source -out /tmp/encoderEmb_deen_2017_mix_src -model ../data/mt_model/model_ende -type encoder_hidden
python ./get_vector_from_sentence/get_embeddings.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2017/data_sys_out -out /tmp/encoderEmb_deen_2017_mix_sys -model ../data/mt_model/model_ende -type encoder_hidden

echo '###########################'
echo '#### decoder embedding ####'
echo '###########################'
echo '***** mixture-3  ******'
echo '---- 2015 ----'
python ./get_vector_from_sentence/get_embeddings.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2015/data_reference.de-en -out /tmp/decoderEmb_deen_2015_mix_ref -model ../data/mt_model/model_deen -type decoder_mixture
python ./get_vector_from_sentence/get_embeddings.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2015/data_source.de-en  -out /tmp/decoderEmb_deen_2015_mix_src -model ../data/mt_model/model_deen -type decoder_mixture
python ./get_vector_from_sentence/get_embeddings.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2015/data_sys_out.de-en -out /tmp/decoderEmb_deen_2015_mix_sys -model ../data/mt_model/model_deen -type decoder_mixture
echo '---- 2016 ----'
python ./get_vector_from_sentence/get_embeddings.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2016/DAseg.newstest2016.reference.de-en -out /tmp/decoderEmb_deen_2016_mix_ref -model ../data/mt_model/model_deen -type decoder_mixture
python ./get_vector_from_sentence/get_embeddings.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2016/DAseg.newstest2016.source.de-en -out /tmp/decoderEmb_deen_2016_mix_src -model ../data/mt_model/model_deen -type decoder_mixture
python ./get_vector_from_sentence/get_embeddings.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2016/DAseg.newstest2016.mt-system.de-en -out /tmp/decoderEmb_deen_2016_mix_sys -model ../data/mt_model/model_deen -type decoder_mixture
echo '---- 2017 ----'
python ./get_vector_from_sentence/get_embeddings.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2017/data_ref -out /tmp/decoderEmb_deen_2017_mix_ref -model ../data/mt_model/model_deen -type decoder_mixture
python ./get_vector_from_sentence/get_embeddings.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2017/data_source -out /tmp/decoderEmb_deen_2017_mix_src -model ../data/mt_model/model_deen -type decoder_mixture
python ./get_vector_from_sentence/get_embeddings.py -src /Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_da_de.en/extracted_data_2017/data_sys_out -out /tmp/decoderEmb_deen_2017_mix_sys -model ../data/mt_model/model_deen -type decoder_mixture


