import numpy as np
from keras.preprocessing import sequence

training_set_IDs = [303, 304, 305, 310, 312, 313, 315, 316, 317, 318, 319, 320, 321, 322, 324, 325, 326, 327, 328, 330, 333, 336, 338, 339, 340, 341, 343, 344, 345, 347, 348, 350, 351, 352, 353, 355, 356, 357, 358, 360, 362, 363, 364, 366, 368, 369, 370, 371, 372, 374, 375, 376, 379, 380, 383, 385, 386, 391, 392, 393, 397, 400, 401, 409, 412, 414, 415, 416, 419, 423, 425, 426, 427, 428, 429, 430, 433, 434, 437, 441, 443, 445, 446, 447, 448, 449, 454, 455, 456, 457, 459, 463, 464, 468, 471, 473, 474, 475, 478, 479, 485, 486, 487, 488, 491]

X = []

for ID in training_set_IDs:
    print(ID, end='\r')
    location = '/data/chercheurs/qureshi191/preprocessed_data/training_data/visual/eye_gaze/individual/'+str(ID)+'_eye_gaze.npy'
    a = np.load(location).T
    X.append(sequence.pad_sequences(a, maxlen=6500, dtype='float32', padding='pre').T.tolist())

X = np.array(X)

np.save('/data/chercheurs/qureshi191/preprocessed_data/training_data/visual/eye_gaze/eye_gaze.npy', X)