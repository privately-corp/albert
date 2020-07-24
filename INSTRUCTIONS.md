# Pretraining Instructions

There are three corpora under data/de_corpus

- DE Wiki: `de_wiki/` (5.4 GB, already split into 54 chunk files under chunks/)
- OpenLegalData: `OpenLegalData/` (1.8 GB, already split into 18 chunk files under chunks/)
- EUbookshop: `EUbookshop/` (2.2 GB, not yet split as can't divide corpus into documents)

## 1. Split corpora into chunks

To split a corpus, use the following command:

```
split -n <N_PARTS> -d <CORPUS_FILE>
```

where `<N_PARTS>` is the factor needed to split the corpus into \~100 MB chunks.

## 2. Train Tokenizer

First concatenate the corpora into a single corpus file under data/de_corpus:

```
cat de_wiki/corpus.txt OpenLegalData/corpus.txt <etc> | shuffle > corpus.txt
```

Train the sentencepiece tokenizer by running this bash script:

```
./train_tokenizer.sh
```

The output files are `30k-clean.model` and `30k-clean.vocab` (we only need the `model` file going forward).

## 3. Create Pretraining Data

To create the pretraining records used to train the model, run the following command for each data corpus:

```
./create_data.sh
```

This operation is *very* computationally expensive (it's the reason we need to split the corpora into 100 MB chunks). The underlying python script, `create_pretraining_data.py`, is single-threaded so I've written the script in such a way that it loads and processes multiple file chunks at a time for creating the records.

The logic is already in place for the 54 wikipedia corpus chunks. Each process seems to take \~12 GB of ram which is why I've restricted it to do 4 chunks at a time, which seems to work. In order to process the other corpora, just adapt the looping logic. E.g. for the 18 OpenLegalData chunks, run the first outer loop over `{0..4}` and the 2nd outer loop over `{16..17}`.

The `split` tool seems to generate chunk files with names of the form `x<N>`, where `<N>` can have leading zeros - my script doesn't take this into account, so you'll need to manually rename the chunks with leading zeros and for them to be read in. The preprocessed files are output to a subdir called `preproc/` under each corpus dir.

## 4. Upload pretraining data to Google Cloud

The `gsutil` tool is already installed and is used to copy files to/from the relevant bucket. It might make