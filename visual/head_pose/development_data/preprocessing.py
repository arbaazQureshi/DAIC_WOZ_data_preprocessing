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

	data = pd.read_csv('/data/chercheurs/qureshi191/raw_data/'+str(ID)+'_P/'+str(ID)+'_CLNF_pose.txt', sep=', ')
	
	data = data[(data['timestamp']*10)%3 == 0][:]   #subsampling the data (frame_gap = 0.3 seconds)
	data = data[data['success'] == 1][:]

	no_of_frames = data.shape[0]

	if(max_frames < no_of_frames):
		max_frames = no_of_frames

	if(min_frames > no_of_frames):
		min_frames = no_of_frames

	data_array = data.values
	data_array = data_array[:, 4:]
	data_array = preprocessing.scale(data_array)

	np.save('/data/chercheurs/qureshi191/preprocessed_data/development_data/visual/head_pose/individual/'+str(ID)+'_head_pose.npy', data_array)

print(max_frames, min_frames)