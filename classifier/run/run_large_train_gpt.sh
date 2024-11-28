DATE=$(date +%Y_%m_%d)/$(date +%H_%M_%S)
MODEL=t5-large
DATASET_NAME=hotpotqa
GPU=7

# Absolute Paths to binary HotpotQA dataset
TRAIN_FILE="/home/ash/Adaptive-RAG/classifier/data/hotpotqa/binary/hotpotqa_train.jsonl"
VALID_FILE="/home/ash/Adaptive-RAG/classifier/data/hotpotqa/binary/hotpotqa_subset.jsonl"
PREDICT_FILE="/home/ash/Adaptive-RAG/classifier/data/hotpotqa_predictions/predict.json"

for EPOCH in 35 40
do
    # Train
    TRAIN_OUTPUT_DIR="/home/ash/Adaptive-RAG/outputs/${DATASET_NAME}/model/${MODEL}/epoch/${EPOCH}/${DATE}/train"
    mkdir -p ${TRAIN_OUTPUT_DIR}
    
    CUDA_VISIBLE_DEVICES=${GPU} python run_classifier.py \
        --model_name_or_path ${MODEL} \
        --train_file ${TRAIN_FILE} \
        --question_column question \
        --answer_column answer \
        --learning_rate 3e-5 \
        --max_seq_length 384 \
        --doc_stride 128 \
        --per_device_train_batch_size 32 \
        --output_dir ${TRAIN_OUTPUT_DIR} \
        --overwrite_cache \
        --do_train \
        --num_train_epochs ${EPOCH}
    
    # Validation
    VALID_OUTPUT_DIR="/home/ash/Adaptive-RAG/outputs/${DATASET_NAME}/model/${MODEL}/epoch/${EPOCH}/${DATE}/valid"
    mkdir -p ${VALID_OUTPUT_DIR}
    
    CUDA_VISIBLE_DEVICES=${GPU} python run_classifier.py \
        --model_name_or_path ${TRAIN_OUTPUT_DIR} \
        --validation_file ${VALID_FILE} \
        --question_column question \
        --answer_column answer \
        --max_seq_length 384 \
        --doc_stride 128 \
        --per_device_eval_batch_size 100 \
        --output_dir ${VALID_OUTPUT_DIR} \
        --overwrite_cache \
        --do_eval
    
    # Prediction
    PREDICT_OUTPUT_DIR="/home/ash/Adaptive-RAG/outputs/${DATASET_NAME}/model/${MODEL}/epoch/${EPOCH}/${DATE}/predict"
    mkdir -p ${PREDICT_OUTPUT_DIR}
    
    CUDA_VISIBLE_DEVICES=${GPU} python run_classifier.py \
        --model_name_or_path ${TRAIN_OUTPUT_DIR} \
        --validation_file ${PREDICT_FILE} \
        --question_column question \
        --answer_column answer \
        --max_seq_length 384 \
        --doc_stride 128 \
        --per_device_eval_batch_size 100 \
        --output_dir ${PREDICT_OUTPUT_DIR} \
        --overwrite_cache \
        --do_eval
done
