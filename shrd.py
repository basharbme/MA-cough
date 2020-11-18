# Copyright (c) 2020, Sibel Inc.
# 
# All rights reserved.
# 
# Copyright 2018-2020 [Sibel Inc.], All Rights Reserved.
# NOTICE: All information contained herein is, and remains the property of SIBEL
# INC. The intellectual and technical concepts contained herein are proprietary 
# to SIBEL INC and may be covered by U.S. and Foreign Patents, patents in 
# process, and are protected by trade secret or copyright law. Dissemination of
# this information or reproduction of this material is strictly forbidden unless
# prior written permission is obtained from SIBEL INC. Access to the source code
# contained herein is hereby forbidden to anyone except current SIBEL INC 
# employees, managers or contractors who have executed Confidentiality and 
# Non-disclosure agreements explicitly covering such access.
# The copyright notice above does not evidence any actual or intended 
# publication or disclosure of this source code, which includes information that
# is confidential and/or proprietary, and is a trade secret, of SIBEL INC.
# ANY REPRODUCTION, MODIFICATION, DISTRIBUTION, PUBLIC PERFORMANCE, OR PUBLIC
# DISPLAY OF OR THROUGH USE OF THIS SOURCE CODE WITHOUT THE EXPRESS WRITTEN
# CONSENT OF COMPANY IS STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE
# LAWS AND INTERNATIONAL TREATIES. THE RECEIPT OR POSSESSION OF THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS TO
# REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE, USE, OR
# SELL ANYTHING THAT IT MAY DESCRIBE, IN WHOLE OR IN PART.

import os
import numpy as np
import pandas as pd
from enum import IntEnum
import struct
import datetime

class DataType(IntEnum):
    IMU_LSM6DSL_MODE_4   = 33 # ACCEL X: 416hz ACCEL Y: 52hz ACCEL Z: 52hz
    TEMP_MAX30205_MODE_4 = 53 #  Temperature .0167 Hz from LSM6DSL
    META_DATA            = 200
    INVALID_MODE         = 255 #  Temperature .016 Hz 
    INVALID_MODE_2       = 0 #  Temperature .016 Hz 

# Sampling Period
IMU_MODE4_PERIOD = 2.348772726754413/4
IMU_FINAL_PERIOD = -1
MAX30205_MODE4 = 5000.0

# Delay
LSM6DSL_SENSOR_DELAY = +99.555

# Accelometer
IMU_MODE3_SCALER = 16384.0

# Page Info
PAGE_BYTE = 2048
PRESCALER_ADAM = 8.0
RTC_FREQ = 32.768

TEMP_LSB = 0.00390625

MAGIC_NUM = 0x37

DATATYPE_INDEX_PER_PAGE = { 
    DataType.IMU_LSM6DSL_MODE_4   : 816,
    DataType.TEMP_MAX30205_MODE_4 : 1020,
    DataType.INVALID_MODE : 0,
    DataType.INVALID_MODE_2 : 0
}

def get_dict_df(file_name, extract_sampling_rate = True):
    file_bytes_array = open(file_name,"rb").read()
    total_size = len(file_bytes_array)
    meta_data = {}

    if(total_size == 0):
        print("File does not exist or there is no content in the file")
        
    if total_size % PAGE_BYTE != 0:
        if total_size % PAGE_BYTE == 26:

            print("Header Info Found")
            meta_data = get_header_info(file_bytes_array[:26])
            file_bytes_array = file_bytes_array[26:]
        else:
            print("File's data size adjusted ", total_size % PAGE_BYTE)
            meta_data = get_header_info(file_bytes_array[:total_size % PAGE_BYTE])
            file_bytes_array = file_bytes_array[total_size % PAGE_BYTE:]
    print("File size is ",total_size)
    
    #----------------- Genearte page arrays ---------------------
    page_array = np.frombuffer(file_bytes_array, dtype = np.uint8).reshape(-1, PAGE_BYTE)
    pages_len = page_array.shape[0]
    print("%d pages found" % pages_len)
    #----------------- Parse Page Info ---------------------
    data_type = page_array[:, 0]

    PRESCALER = PRESCALER_ADAM

    flag = page_array[:, 1]
    data_len = ((page_array[:, 3].astype(np.uint16) << 8) + page_array[:, 2].astype(np.uint16))
    epoch_time = ((page_array[:, 7].astype(np.uint64) << 24) | (page_array[:, 6].astype(np.uint64) << 16) | 
                (page_array[:, 5].astype(np.uint64) << 8) | page_array[:, 4].astype(np.uint64)) * PRESCALER / RTC_FREQ

    raw_array = page_array[:, 8:]
    ret_dict = {}
    #----------------- Parse ACCL ---------------------
    global IMU_FINAL_PERIOD
    accl_page = data_type == DataType.IMU_LSM6DSL_MODE_4
    if(accl_page.sum() > 0):
        if(extract_sampling_rate and np.sum(accl_page) > 10):
            IMU_FINAL_PERIOD = find_sampling_rate(DataType.IMU_LSM6DSL_MODE_4, accl_page, epoch_time)
        else:
            IMU_FINAL_PERIOD = IMU_MODE4_PERIOD
        ret_dict[DataType.IMU_LSM6DSL_MODE_4] = parse_IMU_LSM6DSL_MODE_4(accl_page, raw_array, epoch_time, data_len, flag)
        
    #----------------- Parse TEMP ---------------------
    temp_page = data_type == DataType.TEMP_MAX30205_MODE_4
    if(temp_page.sum() > 0):
        ret_dict[DataType.TEMP_MAX30205_MODE_4] = parse_TEMP_MAX30205_MODE_4(temp_page, raw_array, epoch_time, data_len, flag)


    #----------------- Add Meta Data ------------------
    ret_dict[DataType.META_DATA]= meta_data

    return ret_dict

def get_header_info(header):
    meta = {}
    
    session_id = str(struct.unpack('<Q', header[0:8])[0])
    device_type = header[12]
    epoch_time = struct.unpack('<I', header[8:12])[0]
    start_time_epoch = datetime.datetime.utcfromtimestamp(epoch_time).strftime('%Y-%m-%d %H:%M:%S')

    device_name = ""
    for c in header[14:]:
        if c == 0:
            break
        device_name += (chr(c))

    meta['sessionID'] = session_id
    meta['startSessionTime'] = start_time_epoch
    meta['deviceName'] = device_name
    return meta

def parse_IMU_LSM6DSL_MODE_4(page_ind, page_array, epoch_time, data_len, flag):
    cur_page_array = page_array[page_ind]
    cur_page_array = cur_page_array.reshape(len(cur_page_array),-1, 2)
    accl_epoch = epoch_time[page_ind]

    xy_len = int(cur_page_array.shape[1] / 10)
    x_ind = np.arange(xy_len) * 10 + 7
    del_ind = np.arange(xy_len) * 9 + 7
    y_ind = np.arange(xy_len) * 10 + 8

    z_ind = np.arange(cur_page_array.shape[1])
    z_ind = np.delete(z_ind, x_ind)
    z_ind = np.delete(z_ind, del_ind) #np.delete works by deleting that index, so use del_ind
    accl = np.int16((cur_page_array[:, :, 1].astype(np.uint16) << 8) | cur_page_array[:, :, 0]) / IMU_MODE3_SCALER
    accl_x = accl[:, x_ind].reshape(-1)
    accl_y = accl[:, y_ind].reshape(-1)
    accl_z = accl[:, z_ind].reshape(-1)
    accl_time = np.linspace(accl_epoch - ((len(z_ind) - 1) * IMU_FINAL_PERIOD) + LSM6DSL_SENSOR_DELAY, accl_epoch + LSM6DSL_SENSOR_DELAY, len(z_ind), axis = 1).reshape(-1)

    accl_x_full = np.zeros(len(accl_time)) * np.nan
    accl_y_full = np.zeros(len(accl_time)) * np.nan
    xy_ind = np.arange(int(len(accl_z) / 8)) * 8 + 7
    accl_x_full[xy_ind] = accl_x
    accl_y_full[xy_ind] = accl_y

    # Note: There is constant delay of (864 - (2040 / 2 * 0.8)) * IMU_MODE3_PERIOD when compared with original shrd.py parser
    # accl_df = pd.DataFrame()
    # accl_df['time(ms)'] = accl_time
    # accl_df['xl_x'] = accl_x_full
    # accl_df['xl_y'] = accl_y_full
    # accl_df['xl_z'] = accl_z
    
    accl_df = np.concatenate((accl_time.reshape(-1,1),accl_x_full.reshape(-1,1),accl_y_full.reshape(-1,1),accl_z.reshape(-1,1)),axis=1)


    return accl_df

def parse_TEMP_MAX30205_MODE_4(page_ind, page_array, epoch_time, data_len, flag):
    cur_page_array = page_array[page_ind]
    cur_page_array = cur_page_array.reshape(len(cur_page_array),-1, 2) 
    temp_epoch = epoch_time[page_ind]
    temp_len = data_len[page_ind].sum()

    temp = ((np.int16((cur_page_array[:, :, 1].astype(np.uint16) << 8) | cur_page_array[:, :, 0]) * TEMP_LSB * 10) * 0.1).reshape(-1)[:temp_len]#Temp page is bound by data_len 
    temp_time = np.linspace(temp_epoch - ((cur_page_array.shape[1] - 1) * MAX30205_MODE4), temp_epoch, cur_page_array.shape[1], axis = 1).reshape(-1)[:temp_len]
    try:
        temp_time[-int(temp_len % cur_page_array.shape[1]):] = np.linspace(temp_epoch[-1] - ((temp_len % cur_page_array.shape[1] - 1) * MAX30205_MODE4), temp_epoch[-1], int(temp_len % cur_page_array.shape[1]))
    except:
        pass

    if(temp_epoch[0] == temp_epoch[-1]):
        # if no time sync, use sampling rate as estimate
        temp_time = np.linspace(0, len(temp_time) * MAX30205_MODE4, len(temp_time))

    temp_df = pd.DataFrame()
    temp_df['time(ms)'] = temp_time
    temp_df['temp'] = temp + 25

    return temp_df

def find_sampling_rate(datatype, page_ind, epoch_time):
    cur_epoch = epoch_time[page_ind]
    sorted_time_diff = np.sort(np.diff(cur_epoch))
    filtered_time_diff = sorted_time_diff[len(cur_epoch)//3:-len(cur_epoch)//3]

    data_sampling_rate = np.mean(filtered_time_diff)/DATATYPE_INDEX_PER_PAGE[datatype]
    return data_sampling_rate