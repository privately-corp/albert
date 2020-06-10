# python create_pretraining_data.py \
# 	--input_file data/corpus_debug.txt \
# 	--output_file data/corpus_preproc_debug.tfrecord \
# 	--vocab_file data/30k-clean.vocab \
# 	--spm_model_file data/30k-clean.model \
# 	--do_lower_case False \
# 	--dupe_factor 10 \
# 	--max_seq_length 128

for i in {8..11}
do
	python create_pretraining_data.py \
		--input_file data/corpus_${i}.txt \
		--output_file data/corpus_preproc_${i}.tfrecord \
		--vocab_file data/30k-clean.vocab \
		--spm_model_file data/30k-clean.model \
		--do_lower_case False \
		--dupe_factor 10 \
		--max_seq_length 128 &
done

wait

for i in {12..15}
do
	python create_pretraining_data.py \
		--input_file data/corpus_${i}.txt \
		--output_file data/corpus_preproc_${i}.tfrecord \
		--vocab_file data/30k-clean.vocab \
		--spm_model_file data/30k-clean.model \
		--do_lower_case False \
		--dupe_factor 10 \
		--max_seq_length 128 &
done

wait

for i in {16..19}
do
	python create_pretraining_data.py \
		--input_file data/corpus_${i}.txt \
		--output_file data/corpus_preproc_${i}.tfrecord \
		--vocab_file data/30k-clean.vocab \
		--spm_model_file data/30k-clean.model \
		--do_lower_case False \
		--dupe_factor 10 \
		--max_seq_length 128 &
done

wait

for i in {20..21}
do
	python create_pretraining_data.py \
		--input_file data/corpus_${i}.txt \
		--output_file data/corpus_preproc_${i}.tfrecord \
		--vocab_file data/30k-clean.vocab \
		--spm_model_file data/30k-clean.model \
		--do_lower_case False \
		--dupe_factor 10 \
		--max_seq_length 128 &
done

wait