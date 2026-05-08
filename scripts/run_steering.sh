#! /bin/bash

python steering_esmc_optimization_hybrid_vv.py \
  --data_path test_seq.csv \
  --optimal_layer 2 \
  --property sol \
  --output_file res/seq_optimized.csv \
  --sv_from sv \
  --round 4 \
  --T 2
