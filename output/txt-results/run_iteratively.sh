#!/usr/bin/bash

for i in $(seq 1 $1)
do
    python 1-embedding_evaluation.py > r$i.out 
done
