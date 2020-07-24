import tensorflow as tf
import sentencepiece as spm
import six
import numpy as np
import csv
import tqdm
from sklearn.metrics import classification_report
import unicodedata
import time

SPIECE_UNDERLINE = u"▁".encode("utf-8")

def preprocess_text(inputs, remove_space=True, lower=False):
	"""preprocess data by removing extra space and normalize data."""
	outputs = inputs
	if remove_space:
		outputs = " ".join(inputs.strip().split())

	outputs = unicodedata.normalize("NFKD", outputs)
	outputs = "".join([c for c in outputs if not unicodedata.combining(c)])
	if lower:
		outputs = outputs.lower()

	return outputs

def encode_pieces(sp_model, text, return_unicode=True, sample=False):
	"""turn sentences into word pieces."""

	pieces = sp_model.EncodeAsPieces(text)

	new_pieces = []
	for piece in pieces:
		if len(piece) > 1 and piece[-1] == "," and piece[-2].isdigit():
			cur_pieces = sp_model.EncodeAsPieces(
					six.ensure_binary(piece[:-1]).replace(SPIECE_UNDERLINE, b""))
			if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
				if len(cur_pieces[0]) == 1:
					cur_pieces = cur_pieces[1:]
				else:
					cur_pieces[0] = cur_pieces[0][1:]
			cur_pieces.append(piece[-1])
			new_pieces.extend(cur_pieces)
		else:
			new_pieces.append(piece)

	return new_pieces

MAX_SEQ_LEN = 128

X = []
Y_TRUE = []
with open('data/hate_de/test_sg.tsv') as tsvin:
	for row in csv.reader(tsvin, delimiter='\t'):
		X.append(row[1])
		label = 1 if row[0] == 'hate' else 0
		Y_TRUE.append(label)

X = ["Du bist dumm"]
Y_TRUE = [1]

# X = X[:10]
# Y_TRUE = Y_TRUE[:10]

sp_model = spm.SentencePieceProcessor()
sp_model_ = tf.io.gfile.GFile("30k-clean.model", "rb").read()
sp_model.LoadFromSerializedProto(sp_model_)

interpreter = tf.lite.Interpreter(model_path='1595604818_float32.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# for id in input_details:
# 	print(id['name'])
# print()
# for od in output_details:
# 	print(od['name'])
# exit()

Y_PRED = []
t0 = time.time()
for text in tqdm.tqdm(X):

	text = preprocess_text(text, lower=True)
	split_tokens = encode_pieces(sp_model, text, return_unicode=False)
	split_tokens = split_tokens[0:(MAX_SEQ_LEN - 2)]

	tokens = []
	segment_ids = []
	tokens.append("[CLS]")
	segment_ids.append(0)

	for token in split_tokens:
		tokens.append(token)
		segment_ids.append(0)
	tokens.append("[SEP]")
	segment_ids.append(0)

	input_ids = [sp_model.PieceToId(token) for token in tokens]
	input_mask = [1] * len(input_ids)

	while len(input_ids) < MAX_SEQ_LEN:
		input_ids.append(0)
		input_mask.append(0)
		segment_ids.append(0)

	input_ids = np.array(input_ids, dtype=np.int32).reshape((1, MAX_SEQ_LEN))
	input_mask = np.array(input_mask, dtype=np.int32).reshape((1, MAX_SEQ_LEN))
	segment_ids = np.array(segment_ids, dtype=np.int32).reshape((1, MAX_SEQ_LEN))
	label_ids = np.zeros((1,128), dtype=np.int32)

	print("input_ids:", input_ids)
	print("input_mask:", input_mask)
	print("segment_ids:", segment_ids)

	interpreter.set_tensor(input_details[1]['index'], input_ids)
	interpreter.set_tensor(input_details[0]['index'], input_mask)
	interpreter.set_tensor(input_details[2]['index'], segment_ids)
	interpreter.set_tensor(input_details[3]['index'], label_ids)

	interpreter.invoke()
	# predictions = interpreter.get_tensor(output_details[1]['index'])
	probabilities = interpreter.get_tensor(output_details[0]['index'])

	# print("predictions:", predictions)
	# print("probabilities:", probabilities)

	Y_PRED.append(np.argmax(probabilities))

print(Y_PRED)
t1 = time.time()
print("time: ", (t1 - t0))
print("Y_TRUE:", Y_TRUE)
print("Y_PRED:", Y_PRED)

cr = classification_report(Y_TRUE, Y_PRED, digits=3)
print(cr)