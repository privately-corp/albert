# python create_pretraining_data.py \
# 	--input_file data/de_wiki/x0 \
# 	--output_file data/de_wiki/preproc/corpus_preproc_0.tfrecord \
# 	--vocab_file data/de_wiki/30k-clean.vocab \
# 	--spm_model_file data/de_wiki/30k-clean.model \
# 	--do_lower_case False \
# 	--dupe_factor 10 \
# 	--max_seq_length 128

#for j in {0..12}
#do
# 	echo "processing next batch"
#	for i in $(seq $((j * 4)) $((j * 4 + 3)))
#	do
#		python create_pretraining_data.py \
#			--input_file data/de_wiki/x${i} \
#			--output_file data/de_wiki/preproc/corpus_preproc_${i}.tfrecord \
#			--vocab_file data/de_wiki/30k-clean.vocab \
#			--spm_model_file data/de_wiki/30k-clean.model \
#			--do_lower_case False \
#			--dupe_factor 10 \
#			--max_seq_length 128 &
#	done
#	wait
#done

echo "processing last batch"
for i in {52..53}
do
	python create_pretraining_data.py \
	 	--input_file data/de_wiki/x${i} \
	 	--output_file data/de_wiki/preproc/corpus_preproc_${i}.tfrecord \
	 	--vocab_file data/de_wiki/30k-clean.vocab \
	 	--spm_model_file data/de_wiki/30k-clean.model \
	 	--do_lower_case False \
	 	--dupe_factor 10 \
	 	--max_seq_length 128 &
done
wait
echo "done."
