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
	
	data = pd.read_csv('/data/chercheurs/qureshi191/raw_data/'+str(ID)+'_P/'+str(ID)+'_FORMANT.csv', header=None)

	data = data.values
	a = np.arange(data.shape[0])

	data = preprocessing.scale(data)
	data = data[a%10 == 0]

	if(max_frames < data.shape[0]):
		max_frames = data.shape[0]

	if(min_frames > data.shape[0]):
		min_frames = data.shape[0]

	np.save('/data/chercheurs/qureshi191/preprocessed_data/test_data/acoustic/formant/individual/'+str(ID)+'_formant.npy', data)

print(max_frames, min_frames)