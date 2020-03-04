import os
import sys
import json
import shutil
import numpy as np


from os import environ
environ["KERAS_BACKEND"] = "tensorflow" #must set backend before importing keras
import tensorflow as tf
from keras.models import load_model
import keras.backend as K



#This takes a model saved in the .h5 format and prepares it as a .pb, for use in CMSSW
#Note, this needs to be run on the GPU with the same version of keras as you trained with
K.set_learning_phase(0)

model = load_model('BEST_model.h5')
print (model.inputs)
print (model.outputs)

with K.get_session() as sess:
    outputs = ["dense_20/Softmax"]
    constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), outputs)
    tf.train.write_graph(constant_graph, 'GraphExport/', "constantgraph.pb", as_text=False)
    if os.path.exists('BuilderExport/'): shutil.rmtree('BuilderExport/')
    builder = tf.saved_model.builder.SavedModelBuilder('BuilderExport/')
    builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING])
    builder.save()
