python run_classifier.py \
    --data_dir=data/hate_de \
    --output_dir=models/finetuned/test_1/ \
    --init_checkpoint=models/pretrained/wiki_2/model.ckpt-500000 \
    --albert_config_file=config.json \
    --spm_model_file=30k-clean.model \
    --do_predict \
    --max_seq_length=128 \
    --task_name='hate' \