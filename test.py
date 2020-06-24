import csv
from sklearn.metrics import classification_report

Y_TRUE = []
c = 0
with open("data/hate_de/test_sg.tsv") as file:
	for line in file:
		parts = line.split("\t")
		Y_TRUE.append(parts[0])

Y_PRED = []
with open("models/finetuned/wiki_2_260k/submit_results.tsv") as tsvin:
	for i, row in enumerate(csv.reader(tsvin, delimiter="\t")):
		if i == 0:
			continue
		Y_PRED.append(row[1])

print(len(Y_TRUE))
print(len(Y_PRED))

cr = classification_report(Y_TRUE, Y_PRED, digits=3)
print(cr)