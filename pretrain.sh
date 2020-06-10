python run_pretraining.py \
    --input_file=data/corpus_preproc_1.tfrecord \
    --output_dir=v1/ \
    --albert_config_file=config.json \
    --do_train \
    --do_eval \
    --train_batch_size=64 \
    --eval_batch_size=64 \
    --max_seq_length=128 \
    --max_predictions_per_seq=20 \
    --optimizer='lamb' \
    --learning_rate=.00176 \
    --num_train_steps=64 \
    --num_warmup_steps=3125 \
    --save_checkpoints_steps=5000
    # --init_checkpoint=... \