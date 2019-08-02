import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re

embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")

files = os.listdir('/data/chercheurs/qureshi191/preprocessed_data/test_data/text/individual/')

all_participants = []

for file in files:
	all_participants.append(np.load('/data/chercheurs/qureshi191/preprocessed_data/test_data/text/individual/'+file).tolist())

tf.logging.set_verbosity(tf.logging.ERROR)
maxim = -1
sentence_embeddings = []

with tf.Session() as session:
	session.run([tf.global_variables_initializer(), tf.tables_initializer()])
	for i in range(0, len(files)):
		print(len(files) - i, end = '\r')
		x = session.run(embed(all_participants[i]))
		np.save('/data/chercheurs/qureshi191/preprocessed_data/test_data/text/USE_transformer_embeddings/individual_embeddings/'+files[i], x)
		
		if(maxim < x.shape[0]):
			maxim = x.shape[0]

print(maxim)