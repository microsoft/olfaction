#!/bin/bash

# Array of n_ORs values
#n_ORs_values=(0 5 10 20 50 100 400 845)

n_ORs_values=(0 845)

# Define other fixed parameters
model="GCN_OR"
dataset="GS_LF_OR"
features="canonical"
esm="650m"
mol2prot="mol2prot"
prev="1"
split="random"

# Define other parameters that may need to be changed
or_db="HORDE"
prev_model_loss="weighted_loss" # weighted_loss if pmp weighted, otherwise unweighted_loss
cuda_device="0"  # Replace with your CUDA device ID (0, 1, 2, etc.)
# set this to be different for each model path
pmp="m2or_cross_attention_batch_size_128_layernorm_90_10"

# change what metric to do early stopping based off of
metric="roc_auc_score"
#metric="pr_auc_score"

# cd to general directory 
cd ..

mkdir -p "${prev_model_loss}_HORDE"

# Loop over the n_ORs values
for n_ORs in "${n_ORs_values[@]}"; do
    rp="gs_lf_${n_ORs}_OR_logits_layernorm"
    rp_full="${prev_model_loss}_HORDE_2/${rp}_${metric}"
    echo "Running with n_ORs=${n_ORs}, rp=${rp}, cuda_device=${cuda_device}"

    # Run the Python script with the specified arguments
    CUDA_VISIBLE_DEVICES=$cuda_device python classification_OR_feat_ESM_fix.py \
        --model $model \
        -d $dataset \
        -f $features \
        -esm $esm \
        -mol2prot \
        -n_ORs $n_ORs \
        -prev $prev \
        -pmp $pmp \
        -s $split \
        -rp $rp_full \
        -me $metric \
        -OR_db $or_db \
        --prev_model_loss $prev_model_loss \

done
