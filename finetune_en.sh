# ALBERT_HUB_MODULE_HANDLE="https://tfhub.dev/google/albert_base/3"
INIT_CHECKPOINT=""

EPOCHS=10
TOTAL_LINES=52617
BATCH_SIZE=64
NUM_STEPS=$(("$TOTAL_LINES"/"$BATCH_SIZE"*"$EPOCHS"))
WARMUP_STEPS=$(("$NUM_STEPS"*6/100))
EPOCH_STEPS=$(("$NUM_STEPS"/"$EPOCHS"))

python run_classifier.py \
    --data_dir=data/hate_en \
    --output_dir=models/finetuned/en_v2/ \
    --init_checkpoint=models/pretrained/albert_base_v2/model.ckpt-best \
    --albert_config_file=models/pretrained/albert_base_v2/albert_config.json \
    --spm_model_file=models/pretrained/albert_base_v2/30k-clean.model \
    --do_train \
    --do_eval \
    --do_lower_case True \
    --train_batch_size=$BATCH_SIZE \
    --eval_batch_size=64 \
    --max_seq_length=128 \
    --optimizer='adamw' \
    --task_name='hate' \
    --learning_rate=3e-5 \
    --train_step=$NUM_STEPS \
    --num_warmup_steps=$WARMUP_STEPS  \
    --save_checkpoints_steps=$EPOCH_STEPS \
    --keep_checkpoint_max=$EPOCHS \
    --export_dir=exp

    
    # --do_train \
    # --do_eval \
    # --do_lower_case \
    # --vocab_file=models/albert_base/30k-clean.vocab \
    # --albert_hub_module_handle $ALBERT_HUB_MODULE_HANDLE \
    
    