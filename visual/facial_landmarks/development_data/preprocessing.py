import pandas as pd
import numpy as np
from sklearn import preprocessing

development_set_IDs = [302, 307, 331, 335, 346, 367, 377, 381, 382, 388, 389, 390, 395, 403, 404, 406, 413, 417, 418, 420, 422, 436, 439, 440, 472, 476, 477, 482, 483, 484, 489, 490, 492]

maxFrames = -1

for ID in development_set_IDs:
	
	print(ID, end='\r')

	location = '/data/chercheurs/qureshi191/raw_data/'+str(ID)+"_P/"+str(ID)+'_CLNF_features3D.txt'

	data = pd.read_csv(location, sep=', ')

	data = data[(data['timestamp']*10)%2 == 0][:]	#Downsampling the data (interframe gap = 0.2 seconds)
	data = data[data['success'] == 1][:]			#Discarding all the frames with success flag = 0

	data = data.values
	data = data[:, 4:]

	data = preprocessing.scale(data)

	np.save('/data/chercheurs/qureshi191/preprocessed_data/development_data/visual/facial_landmarks/individual/'+str(ID)+'_facial_landmarks.npy', data)

	no_of_frames = data.shape[0]

	if(maxFrames < no_of_frames):
		maxFrames = no_of_frames

print(maxFrames)