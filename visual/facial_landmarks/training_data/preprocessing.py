import pandas as pd
import numpy as np
from sklearn import preprocessing

training_set_IDs = [303, 304, 305, 310, 312, 313, 315, 316, 317, 318, 319, 320, 321, 322, 324, 325, 326, 327, 328, 330, 333, 336, 338, 339, 340, 341, 343, 344, 345, 347, 348, 350, 351, 352, 353, 355, 356, 357, 358, 360, 362, 363, 364, 366, 368, 369, 370, 371, 372, 374, 375, 376, 379, 380, 383, 385, 386, 391, 392, 393, 397, 400, 401, 409, 412, 414, 415, 416, 419, 423, 425, 426, 427, 428, 429, 430, 433, 434, 437, 441, 443, 445, 446, 447, 448, 449, 454, 455, 456, 457, 459, 463, 464, 468, 471, 473, 474, 475, 478, 479, 485, 486, 487, 488, 491]

maxFrames = -1

for ID in training_set_IDs:
	
	print(ID, end='\r')

	location = '/data/chercheurs/qureshi191/raw_data/'+str(ID)+"_P/"+str(ID)+'_CLNF_features3D.txt'

	data = pd.read_csv(location, sep=', ')

	data = data[(data['timestamp']*10)%2 == 0][:]	#Downsampling the data (interframe gap = 0.2 seconds)
	data = data[data['success'] == 1][:]			#Discarding all the frames with success flag = 0

	data = data.values
	data = data[:, 4:]

	data = preprocessing.scale(data)

	np.save('/data/chercheurs/qureshi191/preprocessed_data/training_data/visual/facial_landmarks/individual/'+str(ID)+'_facial_landmarks.npy', data)

	no_of_frames = data.shape[0]

	if(maxFrames < no_of_frames):
		maxFrames = no_of_frames

print(maxFrames)