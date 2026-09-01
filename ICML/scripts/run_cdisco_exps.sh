#!/bin/bash

for dataset in waterbirds
do
  echo "Running experiment for: $dataset"
  python ../train.py -m base@_global_=$dataset experiment@_global_=cdisco
done