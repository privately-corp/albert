for i in {1..21}
do
	gsutil cp data/corpus_preproc_${i}.tfrecord gs://albert-ds/
done