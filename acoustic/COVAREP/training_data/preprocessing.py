import pandas as pd
import numpy as np
from sklearn import preprocessing

train_set_ID_list = [303, 304, 305, 310, 312, 313, 315, 316, 317, 318, 319, 320, 321, 322, 324, 325, 326, 327, 328, 330, 333, 336, 338, 339, 340, 341, 343, 344, 345, 347, 348, 350, 351, 352, 353, 355, 356, 357, 358, 360, 362, 363, 364, 366, 368, 369, 370, 371, 372, 374, 375, 376, 379, 380, 383, 385, 386, 391, 392, 393, 397, 400, 401, 409, 412, 414, 415, 416, 419, 423, 425, 426, 427, 428, 429, 430, 433, 434, 437, 441, 443, 445, 446, 447, 448, 449, 454, 455, 456, 457, 459, 463, 464, 468, 471, 473, 474, 475, 478, 479, 485, 486, 487, 488, 491]

max_frames = -1
min_frames = 1000000000

for ID in train_set_ID_list:
	
	print(ID, end='\r')
	
	data = pd.read_csv('/data/chercheurs/qureshi191/raw_data/'+str(ID)+'_P/'+str(ID)+'_COVAREP.csv', header=None)
	transcript = pd.read_csv('/data/chercheurs/qureshi191/raw_data/'+str(ID)+'_P/'+str(ID)+'_TRANSCRIPT.csv', sep='\t')

	data = data.values
	transcript = transcript.values
	
	#data[data[:, 1] == 0] = 0.0
	
	transcript = transcript[transcript[:,2] == 'Participant']
	transcript = transcript[:, [0,1]]
	transcript = (transcript*100 + 0.5).astype(int)
	

	participant_speech_features = []
	
	for i in range(transcript.shape[0]):
		
		start_range = transcript[i,0]-15
		end_range = transcript[i,1]+15
			
		#if(end_range - start_range + 1 > 300):
		participant_speech_features = participant_speech_features + data[start_range: end_range+1].tolist()
	
	participant_speech_features = np.array(participant_speech_features)
	participant_speech_features = participant_speech_features[participant_speech_features[:,1] == 1]

	participant_speech_features[:, 0:1] = preprocessing.scale(participant_speech_features[:, 0:1])
	participant_speech_features[:, 2:] = preprocessing.scale(participant_speech_features[:, 2:])

	participant_speech_features = np.hstack((participant_speech_features[:, 0:1], participant_speech_features[:, 2:]))

	a = np.arange(participant_speech_features.shape[0])
	participant_speech_features = participant_speech_features[a%4 == 0]

	no_of_frames = participant_speech_features.shape[0]

	if(max_frames < no_of_frames):
		max_frames = no_of_frames

	if(min_frames > no_of_frames):
		min_frames = no_of_frames

	np.save('/data/chercheurs/qureshi191/preprocessed_data/training_data/acoustic/COVAREP/individual/'+str(ID)+'_COVAREP.npy', participant_speech_features)

print(max_frames, min_frames)