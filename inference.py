import tensorflow as tf
import sentencepiece as spm
import six
import numpy as np
import csv
import tqdm
from sklearn.metrics import classification_report
import unicodedata

SPIECE_UNDERLINE = u"▁".encode("utf-8")

# def printable_text(text):
# 	"""Returns text encoded in a way suitable for print or `tf.logging`."""

# 	# These functions want `str` for both Python2 and Python3, but in one case
# 	# it's a Unicode string and in the other it's a byte string.
# 	if six.PY3:
# 		if isinstance(text, str):
# 			return text
# 		elif isinstance(text, bytes):
# 			return six.ensure_text(text, "utf-8", "ignore")
# 		else:
# 			raise ValueError("Unsupported string type: %s" % (type(text)))
# 	elif six.PY2:
# 		if isinstance(text, str):
# 			return text
# 		elif isinstance(text, six.text_type):
# 			return six.ensure_binary(text, "utf-8")
# 		else:
# 			raise ValueError("Unsupported string type: %s" % (type(text)))
# 	else:
# 		raise ValueError("Not running on Python2 or Python 3?")


# def encode_pieces(sp_model, text, return_unicode=True, sample=False):
# 	"""turn sentences into word pieces."""

# 	if six.PY2 and isinstance(text, six.text_type):
# 		text = six.ensure_binary(text, "utf-8")

# 	if not sample:
# 		pieces = sp_model.EncodeAsPieces(text)
# 	else:
# 		pieces = sp_model.SampleEncodeAsPieces(text, 64, 0.1)
# 	new_pieces = []
# 	for piece in pieces:
# 		piece = printable_text(piece)
# 		if len(piece) > 1 and piece[-1] == "," and piece[-2].isdigit():
# 			cur_pieces = sp_model.EncodeAsPieces(
# 					six.ensure_binary(piece[:-1]).replace(SPIECE_UNDERLINE, b""))
# 			if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
# 				if len(cur_pieces[0]) == 1:
# 					cur_pieces = cur_pieces[1:]
# 				else:
# 					cur_pieces[0] = cur_pieces[0][1:]
# 			cur_pieces.append(piece[-1])
# 			new_pieces.extend(cur_pieces)
# 		else:
# 			new_pieces.append(piece)

# 	# note(zhiliny): convert back to unicode for py2
# 	if six.PY2 and return_unicode:
# 		ret_pieces = []
# 		for piece in new_pieces:
# 			if isinstance(piece, str):
# 				piece = six.ensure_text(piece, "utf-8")
# 			ret_pieces.append(piece)
# 		new_pieces = ret_pieces

# 	return new_pieces

def preprocess_text(inputs, remove_space=True, lower=False):
  """preprocess data by removing extra space and normalize data."""
  outputs = inputs
  if remove_space:
    outputs = " ".join(inputs.strip().split())

  if six.PY2 and isinstance(outputs, str):
    try:
      outputs = six.ensure_text(outputs, "utf-8")
    except UnicodeDecodeError:
      outputs = six.ensure_text(outputs, "latin-1")

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

X = X[:10]
Y_TRUE = Y_TRUE[:10]

sp_model = spm.SentencePieceProcessor()
sp_model_ = tf.io.gfile.GFile("30k-clean.model", "rb").read()
sp_model.LoadFromSerializedProto(sp_model_)

model = tf.saved_model.load('test2/saved_model/1594808041/')

interpreter = tf.lite.Interpreter(model_path='float_model.tflite')
# interpreter = tf.lite.Interpreter(model_path='test2/albert_model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

Y_PRED = []

examples = []

for text in tqdm.tqdm(X):
# for text in X:
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

	# input_ids = [sp_model.PieceToId(printable_text(token)) for token in tokens]
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

	# input_ids = tf.constant(input_ids)
	# input_mask = tf.constant(input_mask)
	# segment_ids = tf.constant(segment_ids)
	# label_ids=tf.constant(np.zeros((1,128), dtype=np.int32))
	
	# # examples.append([input_ids, input_mask, segment_ids, label_ids])
	# # continue

	# example = tf.train.Example()
	# output = model.signatures['serving_default'](
	# 	input_ids=input_ids,
	# 	input_mask=input_mask,
	# 	segment_ids=segment_ids,
	# 	label_ids=label_ids
	# )
	# print("TENSORFLOW MODEL:")
	# print("TENSORFLOW MODEL:\t", output['probabilities'].numpy())
	# print("predictions:", output['predictions'].numpy())
	# print()

	interpreter.set_tensor(input_details[2]['index'], input_ids)
	interpreter.set_tensor(input_details[3]['index'], input_mask)
	interpreter.set_tensor(input_details[1]['index'], segment_ids)
	interpreter.set_tensor(input_details[0]['index'], label_ids)

	interpreter.invoke()
	# logits = interpreter.get_tensor(output_details[0]['index'])
	predictions = interpreter.get_tensor(output_details[0]['index'])
	probabilities = interpreter.get_tensor(output_details[1]['index'])

	# print("logits:", logits)
	# print("TFLITE MODEL:")
	# print("TFLITE MODEL:\t\t", probabilities)
	# print()
	# print("predictions:", predictions)
	# print()
	# Y_PRED.append(np.argmax(prediction, axis=0))
	Y_PRED.append(predictions[0])

print("Y_TRUE:", Y_TRUE)
print("Y_PRED:", Y_PRED)

# np.save('examples.npy', examples)


cr = classification_report(Y_TRUE, Y_PRED, digits=3)
print(cr)