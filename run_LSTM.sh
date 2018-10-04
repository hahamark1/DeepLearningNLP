#!/bin/sh
#PBS -lwalltime=48:00:00
#PBS -lnodes=1:ppn=12
#PBS -lmem=250GB

python $HOME/DeepLearningNLP/train_rnn.py --num_layers 1 --use_LSTM 1 --test_size 200 --save_every 400 --test_every 25 --print_every 25 --n_iters 20000 --data_path $HOME/DeepLearningNLP/data --summary_path $HOME/DeepLearningNLP/summaries_dl4nlt_lstm --checkpoint_path $HOME/DeepLearningNLP/checkpoints_dl4nlt_lstm > $HOME/DeepLearningNLP/final_lstm.txt