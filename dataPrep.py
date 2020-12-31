#We need to sort the various files into folders with the respective classes to make it easier for PyTorch. 
import os,shutil
import pandas as pd 
fPath = '/home/bhargava/Documents/dataSets/militaryAircraft/dataset' #replace with your folder
filesDict = {} #A dictionary will be easier to keep track of the labels
for file in os.listdir(fPath):
	if '.csv' in file:
		data = pd.read_csv(os.path.join(fPath,file))
		label = list(data['class'])[0]
		filesDict[file[:-4]] = label
#Move the files into the respective folders which have to be created prior to this step
for key in filesDict.keys():
	image = key + '.jpg'
	shutil.move(os.path.join(fPath,image),os.path.join(fPath,filesDict[key]))

for key in filesDict.keys():
	file = key + '.csv'
	shutil.move(os.path.join(fPath,file),os.path.join(fPath,filesDict[key]))
del filesDict
