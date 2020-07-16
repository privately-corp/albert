import tensorflow as tf
import os
import numpy as np

# Convert the model.
converter = tf.lite.TFLiteConverter.from_saved_model('test2/saved_model/1594895687/')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_types = [tf.float16]

# examples = np.load('examples.npy')
# print(examples.shape)
# examples = examples[:1]

# def representative_dataset_gen():
#   for i, example in enumerate(examples):
#     # Get sample input data as a numpy array in a method of your choosing.
#     print(i)
#     yield [example[2], example[3], example[1], example[0]]
# converter.representative_dataset = representative_dataset_gen

# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
										# tf.lite.OpsSet.SELECT_TF_OPS]

# converter.inference_input_type = tf.int8  # or tf.uint8
# converter.inference_output_type = tf.int8  # or tf.uint8

# converter.target_spec.supported_types = [tf.float16]
# converter._experimental_new_quantizer = True  # pylint: disable=protected-access
float_model = converter.convert()

with open("1594895687_int8.tflite", "wb") as file:
    file.write(float_model)