#!/bin/bash

for method in cdisco
do
  echo "Running timing analysis for: $method"
  python ICML/train.py --config-name timing_analysis -m base@_global_=fairface timing_analysis@_global_=$method
done
