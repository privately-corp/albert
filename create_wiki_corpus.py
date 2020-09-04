import glob
import nltk
import tqdm

# tokenizer = nltk.data.load('tokenizers/punkt/german.pickle')
tokenizer = nltk.load('sentence_tokenizer.pickle')


subdirs = glob.glob('de_wiki_articles/*')
input_lines = []

print("read files")
for subdir in tqdm.tqdm(subdirs):
	files = glob.glob(subdir + "/*")
	for file in files:
		with open(file, 'r') as f:
			input_lines += [line.strip() for line in f.readlines() if line.strip() != '']

# with open('example.txt', 'r') as f:
# 			input_lines += [line.strip() for line in f.readlines() if line.strip() != '']

output_lines = []
last_line_empty = False

print("process texts")
for input_line in tqdm.tqdm(input_lines):

	if input_line[:4] == '<doc' or input_line == '':
		continue

	if input_line == '</doc>':
		if not last_line_empty:
			output_lines.append('')
			last_line_empty = True
		continue

	sentences = tokenizer.tokenize(input_line)
	for sentence in sentences:
		if len(sentence.split(" ")) >= 5:
			output_lines.append(sentence)
			last_line_empty = False

print("write corpus")
with open('wiki_corpus.txt', 'w') as f:
	for output_line in tqdm.tqdm(output_lines):
		f.write('%s\n' % output_line)


