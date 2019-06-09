#!/bin/bash
set -e

CORPUS_PROCESSED=$1
edim=200
K=2000000
n_min=1
n_max=8
lr=0.01
n_iteration=10
n_negative_sample=10
st=1e-7
ep=1e-7
n_core=48
OUTPUT=$CORPUS_PROCESSED".comp_ngrams.txt"

./scne --corpus_path=$CORPUS_PROCESSED --output_path=$OUTPUT --voca_size=$K --embed_dim=$edim --random_seed=1 --epoch_num=$n_iteration --neg_num=$n_negative_sample --thread_num=$n_core --learning_rate=$lr --sample_rate=0.0001 --power_freq=0.75 --support_threshold=$st --epsilon=$ep --n_max=$n_max --n_min=$n_min
