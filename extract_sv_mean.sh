#! /bin/bash

python sv_extraction/extract_esmc_steering_vec_mean.py \
  --data_path sv_data/sol/edl/sol_edl_seq.csv \
  --optimal_layer 2 \
  --property sol \
  --save_folder sv \
