import numpy as np
from keras.preprocessing import sequence

test_set_IDs = [300, 301, 306, 308, 309, 311, 314, 323, 329, 332, 334, 337, 349, 354, 359, 361, 365, 378, 384, 387, 396, 399, 405, 407, 408, 410, 411, 421, 424, 431, 432, 435, 438, 442, 450, 452, 453, 461, 462, 465, 466, 467, 469, 470, 481]

X = []

for ID in test_set_IDs:
    print(ID, end='\r')
    location = '/data/chercheurs/qureshi191/preprocessed_data/test_data/visual/eye_gaze/individual/'+str(ID)+'_eye_gaze.npy'
    a = np.load(location).T
    X.append(sequence.pad_sequences(a, maxlen=6500, dtype='float32', padding='pre').T.tolist())

X = np.array(X)

np.save('/data/chercheurs/qureshi191/preprocessed_data/test_data/visual/eye_gaze/eye_gaze.npy', X)