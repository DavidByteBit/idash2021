#!/bin/bash

timestamp=$(date +%d%m%Y%H%M%S)
results="results$timestamp.txt"
echo "Results in $results"
echo "experiments idash2021" > $results

n_examples_train=700
n_examples_test=131
n_features=1874
n_epochs=10
batch_size=128
n_times=5

for i in $(seq 1 $n_times)
do
    ./compile.py -R 64 logistic_regression_idash2021 $n_examples_train $n_examples_test $n_features $n_epochs $batch_size >> $results
    Scripts/ring.sh logistic_regression_idash2021-$n_examples_train-$n_examples_test-$n_features-$n_epochs-$batch_size >> $results
#    ./compile.py -R 64 logistic_regression_idash2021 $n_examples_train $n_examples_test $n_features $n_epochs $batch_size
#    Scripts/ring.sh logistic_regression_idash2021-$n_examples_train-$n_examples_test-$n_features-$n_epochs-$batch_size
done




echo "Accuracy $timestamp - $results" >> accuracy.txt
cat $results | grep accuracy >> accuracy.txt
echo "Time $timestamp - $results" >> times.txt
cat $results | grep Time10 >> times.txt

cat accuracy.txt

cat times.txt
