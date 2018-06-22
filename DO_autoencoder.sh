#/bin/bash

# train model
python experiments/train_autoencoder.py -data '/tmp/out.npy' -model 'vae'
python experiments/train_autoencoder.py -data '/tmp/out.npy' -model 'autoencoder'

# transform
python experiments/transform_autoencoder.py -model 'vae' -checkpoint './checkpoints/vae'
python experiments/transform_autoencoder.py -model 'autoencoder' -checkpoint './checkpoints/autoencoder'

# run and test metric model
python experiments/run_expr_rank.py -doc -experiments/plan_c_rank_deen.xml
