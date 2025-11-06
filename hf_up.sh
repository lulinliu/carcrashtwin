#!/usr/bin/env bash
set -euo pipefail

# 1) your repo
REPO_ID="luuuulinnnn/tacc_output_cosmos"

# 2) paths
BASE_DIR="/scratch/10102/hh29499/carcrashtwin"
TMP_DIR="/scratch/10102/hh29499/tmp_hf_upload"
mkdir -p "$TMP_DIR"

cd "$BASE_DIR"

# 3) disable Xet so HF uses the classic upload path
export HF_HUB_ENABLE_XET=0

# 4) make sure the dataset exists (idempotent)
huggingface-cli repo create "$REPO_ID" --type dataset --private -y || true

# 5) pack the three folders
tar -cf "$TMP_DIR/cosmos_out_ft_rw_3k.tar"  cosmos_out_ft_rw_3k
tar -cf "$TMP_DIR/cosmos_out_ft_syn_3k.tar" cosmos_out_ft_syn_3k
tar -cf "$TMP_DIR/cosmos_out_van.tar"       cosmos_out_van

# 6) upload the tars
huggingface-cli upload "$REPO_ID" "$TMP_DIR/cosmos_out_ft_rw_3k.tar"  "cosmos_out_ft_rw_3k.tar"  --repo-type dataset --commit-message "add rw 3k"
huggingface-cli upload "$REPO_ID" "$TMP_DIR/cosmos_out_ft_syn_3k.tar" "cosmos_out_ft_syn_3k.tar" --repo-type dataset --commit-message "add syn 3k"
huggingface-cli upload "$REPO_ID" "$TMP_DIR/cosmos_out_van.tar"       "cosmos_out_van.tar"       --repo-type dataset --commit-message "add van"
