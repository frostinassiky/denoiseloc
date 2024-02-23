set -ex
ckpt_path=$1
eval_split_name=$2

echo $SS
eval_path=data/highlight_${eval_split_name}_release.jsonl
PYTHONPATH=$PYTHONPATH:. python moment_detr/inference.py \
--resume ${ckpt_path} \
--eval_split_name ${eval_split_name} \
--eval_path ${eval_path} --sample_step 1 --snr_scale 2 \
${@:3}
