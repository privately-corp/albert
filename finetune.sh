# ALBERT_HUB_MODULE_HANDLE="https://tfhub.dev/google/albert_base/3"
INIT_CHECKPOINT=""

EPOCHS=20
TOTAL_LINES=65022
BATCH_SIZE=64
NUM_STEPS=$(("$TOTAL_LINES"/"$BATCH_SIZE"*"$EPOCHS"))
WARMUP_STEPS=$(("$NUM_STEPS"*6/100))
EPOCH_STEPS=$(("$NUM_STEPS"/"$EPOCHS"))

python run_classifier.py \
    --data_dir=data/hate_de \
    --output_dir=models/finetuned/wiki_2_260k \
    --init_checkpoint=models/pretrained/wiki_2/model.ckpt-260000 \
    --albert_config_file=config.json \
    --spm_model_file=data/de_wiki/30k-clean.model \
    --do_predict \
    --train_batch_size=$BATCH_SIZE \
    --eval_batch_size=16 \
    --max_seq_length=128 \
    --optimizer='adamw' \
    --task_name='hate' \
    --learning_rate=5e-6 \
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
    
    