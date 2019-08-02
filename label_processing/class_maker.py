import numpy as np
import pandas as pd

def training_set_class_maker():
	
	train_set_ID_list = [303, 304, 305, 310, 312, 313, 315, 316, 317, 318, 319, 320, 321, 322, 324, 325, 326, 327, 328, 330, 333, 336, 338, 339, 340, 341, 343, 344, 345, 347, 348, 350, 351, 352, 353, 355, 356, 357, 358, 360, 362, 363, 364, 366, 368, 369, 370, 371, 372, 374, 375, 376, 379, 380, 383, 385, 386, 391, 392, 393, 397, 400, 401, 409, 412, 414, 415, 416, 419, 423, 425, 426, 427, 428, 429, 430, 433, 434, 437, 441, 443, 445, 446, 447, 448, 449, 454, 455, 456, 457, 459, 463, 464, 468, 471, 473, 474, 475, 478, 479, 485, 486, 487, 488, 491]
	
	labels = pd.read_csv('/data/chercheurs/qureshi191/raw_data/train_split_Depression_AVEC2017.csv')
	labels.set_index('Participant_ID', inplace = True)

	Y = labels['PHQ8_Score'][train_set_ID_list].values
	#X_gender = labels['Gender'][train_set_ID_list].values

	Y_class = []

	for i in range(len(Y)):
		if(0 <= Y[i] < 5):
			Y_class.append([1,0,0,0,0])
		elif(5 <= Y[i] < 10):
			Y_class.append([0,1,0,0,0])
		elif(10 <= Y[i] < 15):
			Y_class.append([0,0,1,0,0])
		elif(15 <= Y[i] < 20):
			Y_class.append([0,0,0,1,0])
		elif(20 <= Y[i] < 25):
			Y_class.append([0,0,0,0,1])

	Y_class = np.array(Y_class)

	np.save('/data/chercheurs/qureshi191/labels/training_set_class_labels.npy', Y_class)



def development_set_class_maker():
	
	dev_set_ID_list = [302, 307, 331, 335, 346, 367, 377, 381, 382, 388, 389, 390, 395, 403, 404, 406, 413, 417, 418, 420, 422, 436, 439, 440, 472, 476, 477, 482, 483, 484, 489, 490, 492]
	
	labels = pd.read_csv('/data/chercheurs/qureshi191/raw_data/dev_split_Depression_AVEC2017.csv')
	labels.set_index('Participant_ID', inplace = True)

	Y = labels['PHQ8_Score'][dev_set_ID_list].values
	#X_gender = labels['Gender'][train_set_ID_list].values

	Y_class = []

	for i in range(len(Y)):
		if(0 <= Y[i] < 5):
			Y_class.append([1,0,0,0,0])
		elif(5 <= Y[i] < 10):
			Y_class.append([0,1,0,0,0])
		elif(10 <= Y[i] < 15):
			Y_class.append([0,0,1,0,0])
		elif(15 <= Y[i] < 20):
			Y_class.append([0,0,0,1,0])
		elif(20 <= Y[i] < 25):
			Y_class.append([0,0,0,0,1])

	Y_class = np.array(Y_class)

	np.save('/data/chercheurs/qureshi191/labels/development_set_class_labels.npy', Y_class)



def test_set_class_maker():
	
	test_set_ID_list = [300, 301, 306, 308, 309, 311, 314, 323, 329, 332, 334, 337, 349, 354, 359, 361, 365, 378, 384, 387, 396, 399, 405, 407, 408, 410, 411, 421, 424, 431, 432, 435, 438, 442, 450, 452, 453, 461, 462, 465, 466, 467, 469, 470, 481]
	
	labels = pd.read_csv('/data/chercheurs/qureshi191/raw_data/full_test_split.csv')
	labels.set_index('Participant_ID', inplace = True)
	
	Y = labels['PHQ_Score'][test_set_ID_list].values
	#X_gender = labels['Gender'][train_set_ID_list].values

	Y_class = []

	for i in range(len(Y)):
		if(0 <= Y[i] < 5):
			Y_class.append([1,0,0,0,0])
		elif(5 <= Y[i] < 10):
			Y_class.append([0,1,0,0,0])
		elif(10 <= Y[i] < 15):
			Y_class.append([0,0,1,0,0])
		elif(15 <= Y[i] < 20):
			Y_class.append([0,0,0,1,0])
		elif(20 <= Y[i] < 25):
			Y_class.append([0,0,0,0,1])

	Y_class = np.array(Y_class)

	np.save('/data/chercheurs/qureshi191/labels/test_set_class_labels.npy', Y_class)

if __name__ == "__main__":
	training_set_class_maker()
	development_set_class_maker()
	test_set_class_maker()
