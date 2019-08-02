import pandas as pd
import numpy as np
from sklearn import preprocessing

dev_set_ID_list = [302, 307, 331, 335, 346, 367, 377, 381, 382, 388, 389, 390, 395, 403, 404, 406, 413, 417, 418, 420, 422, 436, 439, 440, 451, 458, 472, 476, 477, 482, 483, 484, 489, 490, 492]
incomplete_data_ID_list = [342, 394, 398, 460, 373, 444, 451, 458, 480, 402]

for ID in incomplete_data_ID_list:
	if(ID in dev_set_ID_list):
		dev_set_ID_list.remove(ID)

max_frames = -1
min_frames = 1000000000

for ID in dev_set_ID_list:
	
	print(ID, end='\r')
	
	data = pd.read_csv('/data/chercheurs/qureshi191/raw_data/'+str(ID)+'_P/'+str(ID)+'_FORMANT.csv', header=None)

	data = data.values
	a = np.arange(data.shape[0])

	data = preprocessing.scale(data)
	data = data[a%10 == 0]

	if(max_frames < data.shape[0]):
		max_frames = data.shape[0]

	if(min_frames > data.shape[0]):
		min_frames = data.shape[0]

	np.save('/data/chercheurs/qureshi191/preprocessed_data/development_data/acoustic/formant/individual/'+str(ID)+'_formant.npy', data)

print(max_frames, min_frames)