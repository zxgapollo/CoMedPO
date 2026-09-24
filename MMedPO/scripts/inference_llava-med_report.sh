model_base="microsoft/llava-med-v1_5-mistral-7b"
model_path=/path/to/model/checkpoint
model_path_basename=$(basename $model_path)

image_folder=/path/to/dataset/image_folder


question_file=/path/to/test/jsonl_file
answer_file=/path/to/output/jsonl_file


CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$RANDOM inference/llava-med-1.5_report.py \
    --model-base $model_base \
    --model-path $model_path \
    --question-file $question_file \
    --image-folder $image_folder \
    --answers-file $answer_file \
    --temperature 0.2 \






