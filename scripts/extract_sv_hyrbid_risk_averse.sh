#! /bin/bash

python sv_extraction/extract_esmc_sv_hybrid_precision.py \
  --id_data sv_data/sol/edl/sol_source_seq.csv \
  --ood_data sv_data/sol/edl/sol_edl_seq.csv \
  --uncertainty_path sv_data/sol/edl/sol_edl_seq_uncertainty.csv \
  --uncertainty_type epistemic \
  --score_col prob_pos \
  --optimal_layer 2 \
  --property sol \
  --save_folder sv \
