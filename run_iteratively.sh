#!/bin/bash

source bin/activate
for i in $(seq 1 $1)
do
    nohup python 1-embedding_evaluation.py $> r$i.out &
done