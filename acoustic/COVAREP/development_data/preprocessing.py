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

	np.save('/data/chercheurs/qureshi191/preprocessed_data/development_data/acoustic/COVAREP/individual/'+str(ID)+'_COVAREP.npy', participant_speech_features)

print(max_frames, min_frames)