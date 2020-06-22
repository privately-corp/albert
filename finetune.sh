ALBERT_HUB_MODULE_HANDLE="https://tfhub.dev/google/albert_base/3"
INIT_CHECKPOINT=""

python run_classifier.py \
    --data_dir=data/hate_en \
    --output_dir=vf1/ \
    --init_checkpoint=models/albert_base/model.ckpt-best \
    --albert_config_file=models/albert_base/albert_config.json \
    --spm_model_file=models/albert_base/30k-clean.model \
    --do_lower_case \
    --do_train \
    --do_eval \
    --train_batch_size=128 \
    --eval_batch_size=32 \
    --max_seq_length=64 \
    --optimizer='adamw' \
    --task_name='hate' \
    --learning_rate=1e-5 \
    --train_step=1000 \
    --num_warmup_steps=0 \
    --save_checkpoints_steps=200 \
    --export_dir=exp

    
    # --vocab_file=models/albert_base/30k-clean.vocab \
    # --albert_hub_module_handle $ALBERT_HUB_MODULE_HANDLE \
    
    