python run_classifier.py \
	--data_dir=data/hate_de \
	--task_name='hate' \
	--albert_config_file=config.json \
	--spm_model_file=30k-clean.model \
	--init_checkpoint=models/pretrained/wiki_2/model.ckpt-500000 \
	--export_dir='test2' \
    --max_seq_length=128 \
    --do_predict \
	--output_dir=models/finetuned/test_1/ \