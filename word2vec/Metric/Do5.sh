#!/bin/bash

# correct the error in Do.sh
export PATH="/home/jniehues/anaconda3/bin:$PATH"
export LC_CTYPE=en_US.UTF-8
export LC_ALL=en_US.UTF-8

source activate metrics

#python scripts/showStatistics.py

#srun python scripts/collect_rr_data_orderby_sid.py
#srun python scripts/get_valTauLike2.w2v.py 
#python scripts/get_valTauLike2.w2v.py

#srun python scripts/get_valTauLike2.py

#cat tmp/tmp1 | tail -n +2 | awk '{print $2}' > tmp/scores1

#cat tmp/tmp2 | tail -n +2 | awk '{print $2}' > tmp/scores2

#paste tmp/scores1 tmp/scores2 | awk '{if($1 > $2) print "1"; if($2 > $1) print -1; if($2 == $1) print "0"}' > tmp/compare
#paste collect_data/data_scores tmp/compare  | awk '{if($1 != 0){nom+= 1}; sum +=$1*$2}END{print sum,nom,1.0*sum/nom}'


#srun python scripts/collect_rr_data_orderby_sid.py -csv ./data/wmt15-ende.csv -sys data/en-de_2015/sys/ -ref ./data/newstest2015-ende-ref.de -src ./data/newstest2015-ende-src.en -srclang eng -trglang deu

#python scripts/get_valTauLike2.crf.py 
#python scripts/get_valTauLike2.crf3.py 

echo '#############################'
echo '# w2v ende not slim 2016'
echo '#############################'
echo '+++++++++++++++++sum'
python scripts/get_valTauLike2.w2v.py -s1 ../../../data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2016/data_s1 -s2 ../../../data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2016/data_s2 -ref ../../../data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2016/data_ref -scores ../../../data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2016/data_scores -src ../../../data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2016/data_source -mode sum -model /Users/ihuangyiran/Documents/Workplace_Python/MasterArbeit/word2vec/Metric/data/word2vec/german.model
echo '+++++++++++++++++max'
python scripts/get_valTauLike2.w2v.py -s1 ../../../data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2016/data_s1 -s2 ../../../data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2016/data_s2 -ref ../../../data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2016/data_ref -scores ../../../data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2016/data_scores -src ../../../data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2016/data_source -mode max -model /Users/ihuangyiran/Documents/Workplace_Python/MasterArbeit/word2vec/Metric/data/word2vec/german.model

echo '#############################'
echo '# w2v ende not slim 2015'
echo '#############################'
echo '+++++++++++++++++sum'
python scripts/get_valTauLike2.w2v.py -s1 ../../../data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2015/data_s1 -s2 ../../../data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2015/data_s2 -ref ../../../data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2015/data_ref -scores ../../../data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2015/data_scores -src ../../../data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2015/data_source -mode sum -model /Users/ihuangyiran/Documents/Workplace_Python/MasterArbeit/word2vec/Metric/data/word2vec/german.model
echo '+++++++++++++++++max'
python scripts/get_valTauLike2.w2v.py -s1 ../../../data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2015/data_s1 -s2 ../../../data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2015/data_s2 -ref ../../../data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2015/data_ref -scores ../../../data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2015/data_scores -src ../../../data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2015/data_source -mode max -model /Users/ihuangyiran/Documents/Workplace_Python/MasterArbeit/word2vec/Metric/data/word2vec/german.model

echo '#############################'
echo '# w2v ende not slim 2014'
echo '#############################'
echo '+++++++++++++++++sum'
python scripts/get_valTauLike2.w2v.py -s1 ../../../data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2014/data_s1 -s2 ../../../data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2014/data_s2 -ref ../../../data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2014/data_ref -scores ../../../data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2014/data_scores -src ../../../data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2014/data_source -mode sum -model /Users/ihuangyiran/Documents/Workplace_Python/MasterArbeit/word2vec/Metric/data/word2vec/german.model
echo '+++++++++++++++++max'
python scripts/get_valTauLike2.w2v.py -s1 ../../../data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2014/data_s1 -s2 ../../../data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2014/data_s2 -ref ../../../data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2014/data_ref -scores ../../../data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2014/data_scores -src ../../../data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2014/data_source -mode max -model /Users/ihuangyiran/Documents/Workplace_Python/MasterArbeit/word2vec/Metric/data/word2vec/german.model

echo '#############################'
echo '# w2v ende not slim 2013'
echo '#############################'
echo '+++++++++++++++++sum'
python scripts/get_valTauLike2.w2v.py -s1 ../../../data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2013/data_s1 -s2 ../../../data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2013/data_s2 -ref ../../../data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2013/data_ref -scores ../../../data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2013/data_scores -src ../../../data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2013/data_source -mode sum -model /Users/ihuangyiran/Documents/Workplace_Python/MasterArbeit/word2vec/Metric/data/word2vec/german.model
echo '+++++++++++++++++max'
python scripts/get_valTauLike2.w2v.py -s1 ../../../data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2013/data_s1 -s2 ../../../data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2013/data_s2 -ref ../../../data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2013/data_ref -scores ../../../data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2013/data_scores -src ../../../data/MasterArbeit/plan_c_source_rank_en.de/extracted_data_2013/data_source -mode max -model /Users/ihuangyiran/Documents/Workplace_Python/MasterArbeit/word2vec/Metric/data/word2vec/german.model


echo '#############################'
echo '# w2v deen not slim 2016'
echo '#############################'
echo '+++++++++++++++++sum'
python scripts/get_valTauLike2.w2v.py -s1 ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2016/data_s1 -s2 ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2016/data_s2 -ref ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2016/data_ref -scores ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2016/data_scores -src ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2016/data_source -mode sum
echo '+++++++++++++++++max'
python scripts/get_valTauLike2.w2v.py -s1 ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2016/data_s1 -s2 ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2016/data_s2 -ref ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2016/data_ref -scores ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2016/data_scores -src ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2016/data_source -mode max

echo '#############################'
echo '# w2v deen not slim 2015'
echo '#############################'
echo '+++++++++++++++++sum'
python scripts/get_valTauLike2.w2v.py -s1 ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2015/data_s1 -s2 ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2015/data_s2 -ref ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2015/data_ref -scores ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2015/data_scores -src ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2015/data_source -mode sum
echo '+++++++++++++++++max'
python scripts/get_valTauLike2.w2v.py -s1 ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2015/data_s1 -s2 ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2015/data_s2 -ref ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2015/data_ref -scores ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2015/data_scores -src ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2015/data_source -mode max

echo '#############################'
echo '# w2v deen not slim 2014'
echo '#############################'
echo '+++++++++++++++++sum'
python scripts/get_valTauLike2.w2v.py -s1 ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2014/data_s1 -s2 ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2014/data_s2 -ref ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2014/data_ref -scores ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2014/data_scores -src ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2014/data_source -mode sum
echo '+++++++++++++++++max'
python scripts/get_valTauLike2.w2v.py -s1 ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2014/data_s1 -s2 ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2014/data_s2 -ref ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2014/data_ref -scores ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2014/data_scores -src ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2014/data_source -mode max

echo '#############################'
echo '# w2v deen not slim 2013'
echo '#############################'
echo '+++++++++++++++++sum'
python scripts/get_valTauLike2.w2v.py -s1 ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2013/data_s1 -s2 ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2013/data_s2 -ref ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2013/data_ref -scores ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2013/data_scores -src ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2013/data_source -mode sum
echo '+++++++++++++++++max'
python scripts/get_valTauLike2.w2v.py -s1 ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2013/data_s1 -s2 ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2013/data_s2 -ref ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2013/data_ref -scores ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2013/data_scores -src ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2013/data_source -mode max

echo '#############################'
echo '# w2v deen slim 2016'
echo '#############################'
echo '+++++++++++++++++sum'
python scripts/get_valTauLike2.w2v.py -s1 ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2016/data_s1 -s2 ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2016/data_s2 -ref ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2016/data_ref -scores ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2016/data_scores -src ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2016/data_source -mode sum -model data/word2vec/GoogleNews-vectors-negative300-SLIM.bin.gz
echo '+++++++++++++++++max'
python scripts/get_valTauLike2.w2v.py -s1 ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2016/data_s1 -s2 ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2016/data_s2 -ref ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2016/data_ref -scores ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2016/data_scores -src ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2016/data_source -mode max -model data/word2vec/GoogleNews-vectors-negative300-SLIM.bin.gz

echo '#############################'
echo '# w2v deen slim 2015'
echo '#############################'
echo '+++++++++++++++++sum'
python scripts/get_valTauLike2.w2v.py -s1 ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2015/data_s1 -s2 ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2015/data_s2 -ref ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2015/data_ref -scores ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2015/data_scores -src ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2015/data_source -mode sum -model data/word2vec/GoogleNews-vectors-negative300-SLIM.bin.gz
echo '+++++++++++++++++max'
python scripts/get_valTauLike2.w2v.py -s1 ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2015/data_s1 -s2 ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2015/data_s2 -ref ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2015/data_ref -scores ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2015/data_scores -src ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2015/data_source -mode max -model data/word2vec/GoogleNews-vectors-negative300-SLIM.bin.gz

echo '#############################'
echo '# w2v deen slim 2014'
echo '#############################'
echo '+++++++++++++++++sum'
python scripts/get_valTauLike2.w2v.py -s1 ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2014/data_s1 -s2 ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2014/data_s2 -ref ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2014/data_ref -scores ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2014/data_scores -src ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2014/data_source -mode sum -model data/word2vec/GoogleNews-vectors-negative300-SLIM.bin.gz
echo '+++++++++++++++++max'
python scripts/get_valTauLike2.w2v.py -s1 ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2014/data_s1 -s2 ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2014/data_s2 -ref ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2014/data_ref -scores ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2014/data_scores -src ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2014/data_source -mode max -model data/word2vec/GoogleNews-vectors-negative300-SLIM.bin.gz

echo '#############################'
echo '# w2v deen slim 2013'
echo '#############################'
echo '+++++++++++++++++sum'
python scripts/get_valTauLike2.w2v.py -s1 ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2013/data_s1 -s2 ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2013/data_s2 -ref ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2013/data_ref -scores ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2013/data_scores -src ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2013/data_source -mode sum -model data/word2vec/GoogleNews-vectors-negative300-SLIM.bin.gz
echo '+++++++++++++++++max'
python scripts/get_valTauLike2.w2v.py -s1 ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2013/data_s1 -s2 ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2013/data_s2 -ref ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2013/data_ref -scores ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2013/data_scores -src ../../../data/MasterArbeit/plan_c_source_rank_de.en/extracted_data_2013/data_source -mode max -model data/word2vec/GoogleNews-vectors-negative300-SLIM.bin.gz


