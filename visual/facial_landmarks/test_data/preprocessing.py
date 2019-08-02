import pandas as pd
import numpy as np
from sklearn import preprocessing

test_set_IDs = [300, 301, 306, 308, 309, 311, 314, 323, 329, 332, 334, 337, 349, 354, 359, 361, 365, 378, 384, 387, 396, 399, 405, 407, 408, 410, 411, 421, 424, 431, 432, 435, 438, 442, 450, 452, 453, 461, 462, 465, 466, 467, 469, 470, 481]

maxFrames = -1

for ID in test_set_IDs:
	
	print(ID, end='\r')

	location = '/data/chercheurs/qureshi191/raw_data/'+str(ID)+"_P/"+str(ID)+'_CLNF_features3D.txt'

	data = pd.read_csv(location, sep=', ')

	data = data[(data['timestamp']*10)%2 == 0][:]	#Downsampling the data (interframe gap = 0.2 seconds)
	data = data[data['success'] == 1][:]			#Discarding all the frames with success flag = 0

	data = data.values
	data = data[:, 4:]

	data = preprocessing.scale(data)

	np.save('/data/chercheurs/qureshi191/preprocessed_data/test_data/visual/facial_landmarks/individual/'+str(ID)+'_facial_landmarks.npy', data)

	no_of_frames = data.shape[0]

	if(maxFrames < no_of_frames):
		maxFrames = no_of_frames

print(maxFrames)