#!/bin/bash

python process_w2v_for_file.py -src ./raw_data/rr/deen/2015/data_ref -out ./hidden_data/rr/deen/2015/data_ref -model ./models/english/GoogleNews-vectors-negative300.bin --bin
python process_w2v_for_file.py -src ./raw_data/rr/deen/2015/data_s1 -out ./hidden_data/rr/deen/2015/data_s1 -model ./models/english/GoogleNews-vectors-negative300.bin --bin
python process_w2v_for_file.py -src ./raw_data/rr/deen/2015/data_s2 -out ./hidden_data/rr/deen/2015/data_s2 -model ./models/english/GoogleNews-vectors-negative300.bin --bin

python process_w2v_for_file.py -src ./raw_data/rr/deen/2016/data_ref -out ./hidden_data/rr/deen/2016/data_ref -model ./models/english/GoogleNews-vectors-negative300.bin --bin
python process_w2v_for_file.py -src ./raw_data/rr/deen/2016/data_s1 -out ./hidden_data/rr/deen/2016/data_s1 -model ./models/english/GoogleNews-vectors-negative300.bin --bin
python process_w2v_for_file.py -src ./raw_data/rr/deen/2016/data_s2 -out ./hidden_data/rr/deen/2016/data_s2 -model ./models/english/GoogleNews-vectors-negative300.bin --bin


