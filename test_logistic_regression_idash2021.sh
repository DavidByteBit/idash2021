#!/bin/bash

echo "experiments idash2021" > results.txt

for i in $(seq 1 30)
do
    n_examples_train=700
    n_examples_test=131
    n_features=1874
    n_epochs=200
    batch_size=128
    param_report_loss=True

    ./compile.py -R 64 logistic_regression_idash2021 $n_examples_train $n_examples_test $n_features $n_epochs $batch_size $param_report_loss >> results.txt

    Scripts/ring.sh logistic_regression_idash2021-$n_examples_train-$n_examples_test-$n_features-$n_epochs-$batch_size-$param_report_loss >> results.txt
done

cat results.txt | grep accuracy
