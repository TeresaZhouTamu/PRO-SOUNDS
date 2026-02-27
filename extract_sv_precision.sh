#! /bin/bash

python sv_extraction/extract_esmc_sv_precision.py \
  --data_path sv_data/sol/edl/sol_edl_seq.csv \
  --uncertainty_path sv_data/sol/edl/sol_edl_seq_uncertainty.csv \
  --uncertainty_type epistemic \
  --optimal_layer 2 \
  --property sol \
  --save_folder sv \
