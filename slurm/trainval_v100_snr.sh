#!/bin/bash --login
#SBATCH --job-name m-diff
#SBATCH --array=1,2,5
#SBATCH --time=05:59:00
#SBATCH -o slurm/logs/%A_%a.out
#SBATCH -e slurm/logs/%A_%a.err
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem 40GB
##SBATCH --qos=ivul
#SBATCH --account conf-iccv-2023.03.08-ghanembs


echo Loading Anaconda...
module purge
module load gcc/6.4.0
module load cuda/11.1.1
source activate /ibex/ai/home/xum/miniconda3/envs/denoiseloc


echo `hostname`
which python
nvidia-smi

N=$SLURM_ARRAY_TASK_ID
BS=32
SNR=0.$N

sleep $((RANDOM % 60))

###################### train ############################

dset_name=hl
ctx_mode=video_tef
v_feat_types=slowfast_clip
t_feat_type=clip 
results_root=results
exp_id=snr_$SNR

######## data paths
train_path=data/highlight_train_release.jsonl
eval_path=data/highlight_val_release.jsonl
eval_split_name=val

######## setup video+text features
feat_root=features

# video features
v_feat_dim=0
v_feat_dirs=()
if [[ ${v_feat_types} == *"slowfast"* ]]; then
  v_feat_dirs+=(${feat_root}/slowfast_features)
  (( v_feat_dim += 2304 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi
if [[ ${v_feat_types} == *"clip"* ]]; then
  v_feat_dirs+=(${feat_root}/clip_features)
  (( v_feat_dim += 512 ))
fi

# text features
if [[ ${t_feat_type} == "clip" ]]; then
  t_feat_dir=${feat_root}/clip_text_features/
  t_feat_dim=512
else
  echo "Wrong arg for t_feat_type."
  exit 1
fi

#### training
bsz=$BS

set -ex

PYTHONPATH=$PYTHONPATH:. python moment_detr/train.py \
--dset_name ${dset_name} \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--v_feat_dirs ${v_feat_dirs[@]} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dir ${t_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--results_root ${results_root} \
--exp_id ${exp_id} \
--use_sparse_rcnn 0 \
--use_dynamic_conv 1 \
--use_attention 1 \
--snr_scale $SNR \
${@:1}
