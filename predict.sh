python run_classifier.py \
    --data_dir=data/germeval2018 \
    --output_dir=models/finetuned/test_4/ \
    --albert_config_file=config.json \
    --spm_model_file=30k-clean.model \
    --do_predict \
    --max_seq_length=128 \
    --task_name='hate' \