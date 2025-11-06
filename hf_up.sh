#!/usr/bin/env bash
set -euo pipefail

REPO_ID="luuuulinnnn/tacc_output_cosmos"   # ← change this
BASE_DIR="/scratch/10102/hh29499/carcrashtwin"
TMP_DIR="/scratch/10102/hh29499/tmp_hf_upload"
mkdir -p "$TMP_DIR"

cd "$BASE_DIR"

tar -cf "$TMP_DIR/cosmos_out_ft_rw_3k.tar"  cosmos_out_ft_rw_3k
tar -cf "$TMP_DIR/cosmos_out_ft_syn_3k.tar" cosmos_out_ft_syn_3k
tar -cf "$TMP_DIR/cosmos_out_van.tar"       cosmos_out_van

huggingface-cli upload "$REPO_ID" "$TMP_DIR/cosmos_out_ft_rw_3k.tar"  "cosmos_out_ft_rw_3k.tar"  --repo-type dataset
huggingface-cli upload "$REPO_ID" "$TMP_DIR/cosmos_out_ft_syn_3k.tar" "cosmos_out_ft_syn_3k.tar" --repo-type dataset
huggingface-cli upload "$REPO_ID" "$TMP_DIR/cosmos_out_van.tar"       "cosmos_out_van.tar"       --repo-type dataset
