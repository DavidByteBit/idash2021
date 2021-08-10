#!/bin/bash

echo "experiments idash2021" > results.txt

n_examples_train=700
n_examples_test=131
n_features=1874
n_epochs=10
batch_size=128

for i in $(seq 1 1)
do
#    ./compile.py -R 64 logistic_regression_idash2021 $n_examples_train $n_examples_test $n_features $n_epochs $batch_size >> results.txt
#    Scripts/ring.sh logistic_regression_idash2021-$n_examples_train-$n_examples_test-$n_features-$n_epochs-$batch_size >> results.txt
    ./compile.py -R 64 logistic_regression_idash2021 $n_examples_train $n_examples_test $n_features $n_epochs $batch_size
    Scripts/ring.sh logistic_regression_idash2021-$n_examples_train-$n_examples_test-$n_features-$n_epochs-$batch_size
done

cat results.txt | grep accuracy
