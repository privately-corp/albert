# Pretraining & Finetuning Instructions

## Text Corpora

### German

The full German corpus (16 GB) constitutes four separate corpora:

- DE Wiki: `de_wiki/` (5.5 GB)
- OpenLegalData: `open_legal_data/` (1.8 GB)
- EUbookshop: `eu_bookshop/` (2.2 GB)
- Paracrawl: `paracrawl/` (6 GB)

#### DE Wiki

Dumps obtained from: https://dumps.wikimedia.org/dewiki/

Current dump used: `20200720`

Select a dump version and download the multistream xml archive:

```
dewiki-<dump-version>-pages-articles-multistrebunzip2am.xml.bz2 
```

where `<dump-version>` is the timestamp for the relevant dump.

The dump archive can be downloaded using `wget`, e.g.:

```
wget https://dumps.wikimedia.org/dewiki/<dump-version>/dewiki-<dump-version>-pages-articles-multistream.xml.bz2
```

Extract the archive:

```
bunzip2 dewiki-<dump-version>-pages-articles-multistream.xml.bz2
```

Next, we use [wikiextractor](https://github.com/attardi/wikiextractor) to extract the text articles from the xml dump. Install `wikiextractor` using `pip`:


```
pip install wikiextractor
```

And use it to extract the text corpus from the XML dump as follows:

```
python -m wikiextractor.WikiExtractor -o de_wiki_articles dewiki-<dump-version>-pages-articles-multistream.xml
```

preprocess the corpus:

```
python preprocess_corpus.py de_wiki
```

This will output `de_wiki_corpus_preproc.txt`.


#### OpenLegalData

Dumps obtained from: https://openlegaldata.io/research/2019/02/19/court-decision-dataset.html

The dump archive can be downloaded using `wget`, e.g.:

```
wget https://static.openlegaldata.io/dumps/de/2019-02-19_oldp_cases.json.gz
```

Extract the archive:

```
gunzip 2019-02-19_oldp_cases.json.gz
```


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