python run_pretraining.py \
    --input_file=training_corpora/de_wiki/*, training_corpora/open_legal_data/* \
    --output_dir=models/pretrained/v1/ \
    --albert_config_file=config.json \
    --do_train \
    --do_eval \
    --train_batch_size=512 \
    --eval_batch_size=64 \
    --max_seq_length=128 \
    --max_predictions_per_seq=20 \
    --optimizer='lamb' \
    --learning_rate=.00176 \
    --num_train_steps=64 \
    --num_warmup_steps=3125 \
    --save_checkpoints_steps=5000
    # --init_checkpoint=... \