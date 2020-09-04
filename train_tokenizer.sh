spm_train \
--input data/de_corpus/de_corpus_preprocessed.txt \
--model_prefix=30k-clean \
--vocab_size=30000 \
--pad_id=0 --unk_id=1 --eos_id=-1 --bos_id=-1 \
--control_symbols=[CLS],[SEP],[MASK] \
--user_defined_symbols="(,),\",-,.,–,£,€" \
--shuffle_input_sentence=true \
--input_sentence_size=10000000 \
--character_coverage=0.99995 \
--model_type=unigram

# spm_train \
# --input=data/corpus.txt \
# --model_prefix=30k-clean \
# --vocab_size=30000 \
# --character_coverage=1.0 \
# --character_coverage=0.99995 \
# --model_type=unigram \
# --pad_id=0 --unk_id=1 --eos_id=-1 --bos_id=-1 \
# --control_symbols=[CLS],[SEP],[MASK] \

# spm_train \
# --input data/corpus.txt --model_prefix=30k-clean --vocab_size=30000 \
# --pad_id=0 --unk_id=1 --eos_id=-1 --bos_id=-1 \
# --control_symbols=[CLS],[SEP],[MASK] \
# --shuffle_input_sentence=true --input_sentence_size=10000000 \
# --character_coverage=0.99995 --model_type=unigram \
