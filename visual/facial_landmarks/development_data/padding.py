import numpy as np
from keras.preprocessing import sequence

development_set_IDs = [302, 307, 331, 335, 346, 367, 377, 381, 382, 388, 389, 390, 395, 403, 404, 406, 413, 417, 418, 420, 422, 436, 439, 440, 472, 476, 477, 482, 483, 484, 489, 490, 492]

X = []

for ID in development_set_IDs:
    print(ID, end='\r')
    location = '/data/chercheurs/qureshi191/preprocessed_data/development_data/visual/facial_landmarks/individual/'+str(ID)+'_facial_landmarks.npy'
    a = np.load(location).T
    X.append(sequence.pad_sequences(a, maxlen=10000, dtype='float32', padding='pre').T.tolist())

X = np.array(X)

np.save('/data/chercheurs/qureshi191/preprocessed_data/development_data/visual/facial_landmarks/facial_landmarks.npy', X)