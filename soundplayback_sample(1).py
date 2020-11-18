# All rights reserved
# John A. Rogers Group, Simpson Querrey Institute for Bioelectronics, Northwestern University, Evanston, IL 6208, USA

# This code reads one day data and randomly sample events for labeling

import shrd
import numpy as np
from numpy import genfromtxt
import sys
import os
import simpleaudio.functionchecks as fc
import simpleaudio as sa
import math
import pandas as pd
import samplerate
import matplotlib.pyplot as plt
import csv
from scipy.signal import butter, lfilter
from scipy.io.wavfile import write
import copy
import wave


############################### User Set Paramters #######################################
data_dir = './SRAL2020BF/20-06-08-13_17_32_MSCovid0/'

# Directory storing CNN model predictions
predictions_dir = data_dir + 'output/CNN_morl/'

output_dir = data_dir + 'labels/'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

header = ["Onset", " Annotation", " Class "]
activities = ["Cough", "Talk", "Throat", "Laugh", "Motion"]

##########################################################################################
# input: unfiltered wav file
# output: filtered numpy array

def prepare_events(output_file_name):
    Nclass = len(activities) # total number of classes
    predictions_files = os.listdir(predictions_dir) # each file contains results of one hour
    predictions_files.sort()
    
    # loop over all the hours of predictions files, 
    # construct the aggregated event label file
    # save the counter information (e.g. how many coughs, talks, etc.)

    outputArray = pd.DataFrame(columns = header, index = []) # this is the output aggregated array
    counter = np.zeros(Nclass)
    for predictions_file_name in predictions_files:
        #
        if predictions_file_name.endswith('predictions.txt'):
            predictions_raw = pd.read_csv(os.path.join(predictions_dir, predictions_file_name), delimiter=',')
            Nhour = int(predictions_file_name.split('_')[1])
            classes = predictions_raw[header[2]]
            onset = predictions_raw[header[0]]
            
            predictions_raw[header[0]] = onset+3600*(Nhour-1)
            
            outputArray = outputArray.append(predictions_raw, ignore_index = True)

            # count class
            for ii in range(Nclass):
                counter[ii] += np.sum(classes == ii)

    outputArray.to_csv(output_dir+output_file_name, index=False)
    np.savetxt(output_dir+'counter.txt', counter, delimiter=',', fmt='%g')
    return counter
		
def prepare_events_to_label(input_file_name, output_file_name, ncap):
# here ncap = [300, 300, 100, 100, 300] is an array of maximum number of events prepared for manual check
    outputArray = pd.DataFrame(columns = header, index = [])    
    predictions_raw = pd.read_csv(output_dir+input_file_name, delimiter=',')
    classes = predictions_raw[header[2]]

    counter=[]
    Nclass = len(activities) # total number of classes
    for ii in range(Nclass):
        N = np.sum(classes == ii)           # total number in this class
        P = predictions_raw[classes == ii]  # rows of this class
        if N > ncap[ii]:
            P = P.sample(n = ncap[ii])          # random sample rows (using pandas)
        outputArray = outputArray.append(P, ignore_index=True) # append the selected rows to the new output dataframe
        counter.append(len(P))                   # generate the counter for info
    
    outputArray = outputArray.sort_values(by=['Onset'])   # sort according to time stamp
    outputArray.to_csv(output_dir+output_file_name, index=False)   
    np.savetxt(output_dir+'counter_prep.txt', counter, delimiter=',', fmt='%g')
    return outputArray, counter

def load_raw_data(data_dir):
    #data.shrd is the raw acceleration data file stored in cloud
    data = shrd.get_dict_df(data_dir+'data.shrd')
    accel = data[shrd.DataType.IMU_LSM6DSL_MODE_4]
    accel_z = accel[:,3]
    return accel_z


def labelling_main_loop(data_raw, predictions_file_name, output_file_name):
    # data_raw is the raw acceleration data
    # predictions_file is the data_prep_predictions.txt file, which contains randomly sampled events for manual check

	predictions_raw = pd.read_csv(os.path.join(output_dir+predictions_file_name), delimiter=',')
	outputArray = []
	outputArray.append(header)
	plt.ion()

	i = 0
	while i < len(predictions_raw):
		NClass = predictions_raw[' Class '][i]  ## events
		print('\n \n \n--------------------------------------------------------------------------------')
		print("Label ", i , " out of ", len(predictions_raw))
		outputrow = []
		outputrow.append(predictions_raw['Onset'][i])
		timestampToIndex = int(round(predictions_raw['Onset'][i] * 1666))
		timestart = timestampToIndex-800
		timestop  = timestampToIndex+1200
		if timestampToIndex-800<0:
			timestart = 0
		if timestampToIndex+1200> (len(data_raw)-1):
			timestop = len(data_raw)-1
		event = data_raw[timestart : timestop]
		event = event * int(32767 / max(max(event), abs(min(event))))
		converter = 'sinc_best'  # or 'sinc_fastest', ...
		event_resampled = samplerate.resample(event, 5, converter)
		event_normalized = event_resampled.astype(np.int16)
		play_obj = sa.play_buffer(event_normalized, 1, 2, 8000)
		print("Suspected Event: \n \n", NClass, " ", activities[int(NClass)])
		plt.plot(event)
		plt.axvline(x=480, color='r')
		plt.axvline(x=1120, color='r')
		play_obj.wait_done()
		print(" \nCough: 0		Talk: 1		Throat Clear: 2		Laugh: 3	 Motion: 4")
		x = input("If Correct, press enter. To Repeat, press r (then Enter). To save wav file of sound, \n press s (then Enter). To type a comment, press c (then Enter). Otherwise, type correct label number: \n")
		if(x == ""):
			print("Assigned as correct. Moving on. \n")
			outputrow.append(activities[int(NClass)])
			outputrow.append(int(NClass))
			outputArray.append(outputrow)
		elif(x == "s"):
			print("Saving as Wav File... \n")
			wav_fileName = "AudioClip_Event_" + str(i) + ".wav"
			write(os.path.join(data_dir, wav_fileName), 8000, event_normalized)
			print("File Saved. Now Repeating...\n")
			if i >= 0:
				i -= 1
			# outputrow.append(activities[int(NClass)])
			# outputrow.append(int(NClass))
			# outputArray.append(outputrow)
		elif(x == "r"):
			if i >= 0:
				print("Repeating... \n")
				i -= 1
		elif(x == "c"):
			comment = input("Type your comment: \n")
			outputrow.append(activities[int(NClass)])
			outputrow.append(int(NClass))
			outputrow.append(comment)
			outputArray.append(outputrow)
		else:
			print("Changing to ", x, activities[int(x)])
			outputrow.append(activities[int(x)])
			outputrow.append(int(x))
			outputArray.append(outputrow)
		print('--------------------------------------------------------------------------------')
		plt.clf()
		i += 1
	with open(os.path.join(output_dir,output_file_name), 'w', newline='') as f:
	    csv.writer(f, delimiter=',').writerows(outputArray)

	print("\n \n Confusion Matrix: \n \n ")
	actual_final = np.array(outputArray)[:, 2]
	pred_final = np.array(predictions_raw)[:, 2] 
	y_actu = pd.Series(actual_final,  name='Actual')
	y_pred = pd.Series(pred_final, name='Predicted')
	df_confusion = pd.crosstab(y_actu, y_pred)
	print(df_confusion)

# Main loop starts
#%% 1. collect all label in the directory (predictions_dir - one day data)

output_file_name = 'data_all_predictions.txt'
counter = prepare_events(output_file_name)
print(counter)

#%% 2. prepare events
input_file_name = 'data_all_predictions.txt' # file containing all events and corresponding CNN predictions
output_file_name = 'data_prep_predictions.txt' # file containing only randomly sampled events and corresponding CNN predictions
ncap = [100, 100, 100, 100, 100] # an array of maximum number of events prepared for manual check
outputArray, counter2 = prepare_events_to_label(input_file_name, output_file_name, ncap)

#%% 3. labeling
print("Ground Truth Labeling Program, Covid Sensor Project \n The purpose of this program is to quickly and accurately label patient data from the covid sensors deployed by \n the Rogers Lab. To start, first edit the directories for the data and prediction files, and make sure \n they follow the correct naming specifications \n \n ")

# load data from shrd file
data_raw = load_raw_data(data_dir)

# labeling
predictions_file_name = 'data_prep_predictions.txt'
output_file_name = 'data_prep_predictions_new.txt'

labelling_main_loop(data_raw, predictions_file_name, output_file_name)
print("Program Complete.")

# %%
