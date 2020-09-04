import csv
from sklearn.metrics import classification_report

Y_TRUE = []
c = 0
# with open("data/hate_de/test_sg.tsv") as file:
with open("data/germeval2018/test.tsv") as file:
	for line in file:
		parts = line.split("\t")
		Y_TRUE.append(parts[0])

Y_PRED = []
with open("models/finetuned/test_4/submit_results.tsv") as tsvin:
	for i, row in enumerate(csv.reader(tsvin, delimiter="\t")):
		if i == 0:
			continue
		Y_PRED.append(row[1])

# Y_TRUE = Y_TRUE[:10]
# Y_PRED = Y_PRED[:10]

Y_TRUE = [1 if y == 'hate' else 0 for y in Y_TRUE]
Y_PRED = [1 if y == 'hate' else 0 for y in Y_PRED]

# print("Y_TRUE:", Y_TRUE)
# print("Y_PRED:", Y_PRED)


cr = classification_report(Y_TRUE, Y_PRED, digits=3)
print(cr)