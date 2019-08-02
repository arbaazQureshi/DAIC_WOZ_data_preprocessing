import pandas as pd
import numpy as np
from sklearn import preprocessing

test_set_ID_list = [300, 301, 306, 308, 309, 311, 314, 323, 329, 332, 334, 337, 349, 354, 359, 361, 365, 378, 384, 387, 396, 399, 405, 407, 408, 410, 411, 421, 424, 431, 432, 435, 438, 442, 450, 452, 453, 461, 462, 465, 466, 467, 469, 470, 481]
incomplete_data_ID_list = [342, 394, 398, 460, 373, 444, 451, 458, 480, 402]

for ID in incomplete_data_ID_list:
	if(ID in test_set_ID_list):
		test_set_ID_list.remove(ID)

max_frames = -1
min_frames = 1000000000

for ID in test_set_ID_list:

	print(ID, end='\r')

	data = pd.read_csv('/data/chercheurs/qureshi191/raw_data/'+str(ID)+'_P/'+str(ID)+'_CLNF_AUs.txt', sep=', ')
	
	data = data[(data['timestamp']*10)%3 == 0][:]   #subsampling the data (frame_gap = 0.3 seconds)
	data = data[data['success'] == 1][:]

	no_of_frames = data.shape[0]

	if(max_frames < no_of_frames):
		max_frames = no_of_frames

	if(min_frames > no_of_frames):
		min_frames = no_of_frames

	data_array = data.values
	data_array = data_array[:, 4:]

	data_array[:, 0:14] = preprocessing.scale(data_array[:, 0:14])

	np.save('/data/chercheurs/qureshi191/preprocessed_data/test_data/visual/action_units/individual/'+str(ID)+'_action_units.npy', data_array)

print(max_frames, min_frames)