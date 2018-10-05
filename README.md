# Sentiment Analysis using Convolutions, a baseline comparison.

This repository is the outcome of a project for the course Deep Learning for Natural Language Technology, given by Christof Monz.
The focus of this project was to do Sentiment Analysis using both Recurrent and Convolutional methods. The findings are given in the [report](Report.pdf), furthermore a [presentation](https://docs.google.com/presentation/d/1lNNML0f08YQIsbqO1tAwec2n03Ql-FXPKVFRHz3aCpk/edit?usp=sharing) was given.

To recreate the outcomes of the project one can use the 3 bash files: run_RNN.sh, run_LSTM.sh and run_SCNN.sh. Further hyperparameter tuning could be performed by changing the parameters in the last line of these files.

The dataset used can be found in the data folder. This is a txt file which was given by the [Twitter Sentiment Dataset](http://crowdsourcing-class.org/assignments/downloads/pak-paroubek.pdf) found on [Github](https://github.com/satwantrana/CharSCNN/blob/master/tweets_clean.txt)

We performed the training of the different models on a supercomputer provided by SURF-SARA. This might mean that you cannot obtain the same results locally.
