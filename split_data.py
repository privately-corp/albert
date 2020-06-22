import math

print("reading corpus")
with open("data/de_wiki/corpus.txt", 'r') as file:
	lines = [line.strip() for line in file.readlines()]

print("corpus size", len(lines))
n_chunks = 54
chunks = []
chunk_size = math.ceil(len(lines)/n_chunks)
for i in range(n_chunks):
	print("from:", i*chunk_size, "to:", (i+1)*chunk_size)
	chunk = lines[i*chunk_size: (i+1)*chunk_size]
	chunks.append(chunk)
	print("chunk size:", len(chunk))

for i, chunk in enumerate(chunks):
	print("writing chunk file:", i)
	with open("data/de_wiki/corpus_{}.txt".format(i), 'w') as file:
		for line in chunk:
			file.write("{}\n".format(line))