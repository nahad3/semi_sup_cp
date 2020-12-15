import torch
import math
import os
import numpy as np
import random
from torch.utils.data import  random_split, DataLoader
from load_series import SeriesDataset, Read_signal_and_cp, Get_seg_data
import  matplotlib.pyplot as plt
import mmd_util
from nn_models import  RNNSeriesEncoding
from optim import Optim
from pairs_from_cp import get_pairs_from_cp
from scipy.signal import find_peaks
import scipy.io as sio

#file_path = './../data_files/human_activity/mat_files/Train_scaled_0_1_ActivTraker2011.mat'
#file_path ='./../data_files/gen_simulations/mat_files/mackay_switch/check_cp_switch_mackay.mat'
#file_path = './../data_files/gen_simulations/mat_files/ar_process/ar_train_labels.mat'
#file_path = './../data_files/gen_simulations/mat_files/mackay_switch/noisy_switch_mackay_series_train.mat'
#file_path = './../data_files/gen_simulations/mat_files/mackay_switch/switch_mackay_series_train.mat'
#file_path  ='./../data_files/gen_simulations/mat_files/ar_coeffs_change/ar_signal_train.mat'
#file_path =  './../data_files/real_world_data/HCI_continuous/HCI_continuous.mat'
#file_path = './../data_files/gen_simulations/mat_files/mackay_switch/switch_mackay.mat'
#file_path =  './../../cp_files_from_cpu_machine//hci_sample_same.mat'
#file_path =  './../data_files/real_world_data/HCI_continuous/hci_cp_augmented.mat'
#file_path =  './../data_files/real_world_data/HCI_continuous/hci_cp_untrunc_guided.mat'
#file_path =  './../data_files/real_world_data/HCI_continuous/hci_cp_untrunc_guided_raw_rescaled01.mat'
#file_path = './../data_files/real_world_data/HCI_continuous/HCI_synth_1.mat'
#file_path = './../data_files/real_world_data/PAMAP2/user1.mat'
#file_path = './../data_files/real_world_data/HCI_continuous/hci_cp_untrunc_freehand_raw_rescaled01.mat'
file_path = './../data_files/real_world_data/actitracker/Train_scaled_0_1_ActivTraker2011.mat'

extracted_segs_path = './../data_files/real_world_data/HCI_continuous/labelled_segmented_freehand_01_train_HCI.mat'

#cp_file_path = './../data_files/real_world_data/HCI_continuous/cp_detected_hci_cp_untrunc_freehand_raw_rescaled01.mat'

cuda = 1

if torch.cuda.is_available():
    if not cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

p_win = 500
f_win = 500


#cp_data = Read_signal_and_cp(cp_file_path)

#X_w_cp ,cp_indices = cp_data.read_series()

series_data = SeriesDataset(file_path, p_win, f_win)

#x_segs = Get_seg_data(extracted_segs_path)

#X_segs = x_segs.return_segs()

feats, feats_s01 ,labels = series_data.return_series()
n,d = feats.shape




get_cps_from_long_sequence = 1
aug_sequence = 0

def plot_cps(signal,labels,c_stat, change_indices):
    'Function to plot signal'
    fig,axs = plt.subplots(3)


    axs[0].plot(signal,label = 'Signal')
    axs[0].set_ylabel('Sequence Values')
    axs[0].set_xlabel('Time')
    axs[0].set_title('Sequence')
    axs[1].plot(c_stat,label = 'Change Statistic')
    axs[1].scatter(change_indices, c_stat[change_indices], label = 'Change detected')
    axs[1].set_xlabel('Time')
    axs[1].set_title('Change statistic')
    axs[1].set_ylabel('Change Statistic')
    #axs[1].legend()
    axs[2].plot(labels)
    plt.show()



def check_cp_at_index(X,index,thresh_sim,thresh_dissim):
    '''Function to check if a change point in Signal X around the provided index'''
    p_win = 800
    f_win = 800
    win_pair = 600
    buff = 30
    cp = 0
    #X_temp = X[index - 2*p_win : index+2*p_win,:]
    X_temp = X
    #Y_temp = Y[index - 2*p_win : index+2*p_win,:]
    mmd = np.zeros(len(X_temp))
    var_array = np.zeros(len(X_temp))
    for i in range(p_win, len(X_temp)-f_win):
        past_np_median = X_temp[i - p_win:i, :]
        var_array[i] = np.var(np.mean(X_temp[i - p_win:i, :], axis=1))
        past = torch.from_numpy(X_temp[i - p_win: i, :]).cuda().float()
        past = past.unsqueeze(0)

        fut = torch.from_numpy(X_temp[i: i + f_win, :]).cuda().float()
        fut = fut.unsqueeze(0)
        # labels = labels[ i - p_win : i + f_win]

        sigma_list = mmd_util.median_heuristic(past_np_median, beta=5 )
        sigma_var = torch.FloatTensor(sigma_list).cuda()

        mmd[i] = mmd_util.batch_mmd2_loss(past, fut, sigma_var).cpu().numpy()[0]
    mmd[-100: ] =0
    cp_mag = max(mmd)
    change_indices = np.argmax(mmd)
    #plot_cps(X_temp, 0 , mmd, change_indices)
    sim = 0
    X_pair_1 = X[index - win_pair - buff: index - buff, :]
    X_pair_2 = X[index + buff: index + buff + win_pair, :]
    X1_var = np.median(np.var(X_pair_1[0:500,:], axis=0))
    X2_var = np.median(np.var(X_pair_2[-300:,:], axis=0))
    if X1_var < 0.005 or X2_var < 0.005:  #'For 0 and 0 case '
        thresh_dissim =  2


    if max(mmd[:i-0]) > thresh_dissim:
        sim = -1
    elif  max(mmd) < thresh_sim:
        sim = 1
    else:
        sim = 0


    return  sim


def gel_all_cps_in_sequence(X,window_size,beta_val,threshold_cp,labels):
    '''function to get cps in a long sequence'''
    mmd = np.zeros(len(X))
    var_array_1 = np.zeros(len(X))
    var_array_2 = np.zeros(len(X))
    for i in range(window_size, len(X) - window_size):
        past_np_median = X[i - window_size : i , :]
        var_array_1[i] = np.median(np.var(X[i - window_size:i, :], axis=0))
        past = torch.from_numpy(X[i - window_size: i, :]).cuda().float()
        past = past.unsqueeze(0)

        fut = torch.from_numpy(X[i: i + window_size, :]).cuda().float()
        fut = fut.unsqueeze(0)
        # labels = labels[ i - p_win : i + f_win]

        #sigma_list = mmd_util.median_heuristic(past_np_median, beta= beta_val)
        sigma_list = [0.0001 , 0.01,0.1,1,10]
        sigma_var = torch.FloatTensor(sigma_list).cuda()
        var_array_2[i] = np.median(np.var(X[ i : i + window_size, :], axis=0))
        mmd[i] = mmd_util.batch_mmd2_loss(past, fut, sigma_var).cpu().numpy()[0]
    # mmd[var_array_1 < 0.1] = 0
    # mmd[var_array_2 < 0.1] = 0
    #change_indices = find_peaks(mmd  , distance= window_size, height=  threshold_cp)[0] 'for mackay glass'
    change_indices = find_peaks(mmd, distance=900, height=0.0250)[0] #for acti tracker
    plot_cps(X, labels, mmd, change_indices)
    '''
    sio.savemat('./../data_files/gen_simulations/mat_files/mackay_switch/cp_detected_switch_mackay.mat',
                {'X': X, 'Y': labels,
                 'Cp_stat': mmd, 'var_arr_1': var_array_1,'var_arr_2': var_array_2,
                 'c_indices': change_indices})
                 '''

    return change_indices


def augment_change_sequence(X,cp_indices,buff,pair_length,rep_times):
    'Function to augment sequence to get similar arrays '
    cp_indices = cp_indices.reshape(-1)
    cp_indices = np.insert(cp_indices, 0, buff)
    X_sim = np.array([])
    X1_dissim = np.array([])
    X2_dissim = np.array([])
    X1_sim = np.array([])
    X2_sim = np.array([])

    X_segments = np.array([])

    for i in range(1,len(cp_indices) -2):
        X1_dis_temp = X[cp_indices[i] - buff  -pair_length : cp_indices[i] - buff , : ]
        X2_dis_temp = X[cp_indices[i] + buff: cp_indices[i] + pair_length + buff , : ]



        X1_temp = X1_dis_temp
        X2_temp = X2_dis_temp

        X1_temp = np.tile(X1_temp, (2, 1))
        X2_temp = np.tile(X2_temp, (2,1) )

        temp_seg = np.expand_dims(X1_temp, axis = 0)
        X_segments=  np.concatenate((X_segments,temp_seg),axis = 0) if X_segments.size \
            else temp_seg




        if len(X1_temp) > pair_length and len(X2_temp) > pair_length:
            X1_dis_temp = np.expand_dims( X1_temp[0:int((pair_length/2)),:] , axis = 0 )
            X2_dis_temp = np.expand_dims( X2_temp[0:int((pair_length/2)),:] , axis = 0)
            'Getting dissimilar pairs'
            X1_dissim = np.concatenate((X1_dissim, X1_dis_temp), axis=0) if X1_dissim.size \
                else X1_dis_temp
            X2_dissim = np.concatenate((X2_dissim, X2_dis_temp), axis=0) if X2_dissim.size \
                else X2_dis_temp




            'Getting similar pairs if segment long enough'


            X1_sim1 = np.expand_dims(X1_temp[0:int(pair_length/2),:], axis = 0)
            X2_sim1 = np.expand_dims(X1_temp[int(pair_length/2):pair_length,], axis = 0)

            X1_sim2 = np.expand_dims(X2_temp[0:int(pair_length / 2), :], axis = 0)
            X2_sim2 = np.expand_dims(X2_temp[int(pair_length/ 2):pair_length, :] , axis = 0)



            X1_sim = np.concatenate((X1_sim, X1_sim1), axis=0) if X1_sim.size \
                else X1_sim1
            X1_sim = np.concatenate((X1_sim, X1_sim2), axis=0)


            X2_sim = np.concatenate((X2_sim, X2_sim1), axis=0) if X2_sim.size \
                else X2_sim1
            X2_sim = np.concatenate((X2_sim, X2_sim2), axis=0)


    for i in range(0,2000):
        'get paris which are not adjacent'
        (p1,p2) = random.sample(range(1, len(X_segments)), 2)
        print("aumgneting pair iteration:{0}".format(i))
        seg_1 = X_segments[p1]
        seg_2 = X_segments[p2]
        concatenated_signal = np.concatenate((seg_1,seg_2),axis = 0)
        index = len(seg_1)
        sim = check_cp_at_index(concatenated_signal,index,0.10,0.13)
        X1_temp = np.expand_dims(seg_1[index-int(pair_length/2)- buff:index - buff, :], axis = 0 )
        X2_temp = np.expand_dims( seg_2[buff: buff+int(pair_length/2), :] , axis = 0)


        if sim ==0:
            'dissimilar'
            continue
        elif sim == -1:
            X1_dissim = np.concatenate((X1_dissim, X1_temp), axis=0) if X1_dissim.size \
                else X1_temp
            X2_dissim = np.concatenate((X2_dissim, X2_temp), axis=0) if X2_dissim.size \
                else X2_temp
        elif sim == 1:
            X1_sim = np.concatenate((X1_sim, X1_temp), axis=0) if X1_sim.size \
                else X1_temp
            X2_sim = np.concatenate((X2_sim, X2_temp), axis=0) if X2_sim.size \
                else X2_temp
            'similar and concatenate'

    #min_pair = min(X1_sim.shape[0], X2_dissim.shape[0])
    #max_sim = max(X1_sim.shape[0], X2_dissim.shape[0])
    #samp_integers = random.sample(range(0, max_sim), min_pair)
    #X1_sim1 = X1_sim[samp_integers,:,:]
    #X2_sim2 = X2_sim[samp_integers,:,:]

    X1 = np.concatenate((X1_sim,X1_dissim ),axis = 0)
    X2 = np.concatenate((X2_sim, X2_dissim), axis= 0)
    Y1 = np.ones(X1_sim.shape[0])
    Y2 = -1*np.ones(X1_dissim.shape[0])
    Y = np.concatenate((Y1,Y2),axis=0)
    rand_perm = random.sample(range(0,  Y.shape[0] ), Y.shape[0])
    Y = Y[rand_perm]
    X1 = X1[rand_perm,:,:]
    X2 = X2[rand_perm,:,:]
    train_ratio = 0.7

    X1_train = X1[0:int(train_ratio *len(X1)),:,:]
    X2_train = X2[0:int(train_ratio *len(X2)),:,:]
    Y_train = Y[0:int(train_ratio *len(Y))]

    X1_test = X1[int(train_ratio *len(X1)): ,:,:]
    X2_test = X2[int(train_ratio *len(X2)): ,:,:]
    Y_test = Y[int(train_ratio *len(Y)):]

    sio.savemat('./../data_files/real_world_data/HCI_continuous/hci_cp_pairs_train_01.mat', {'X1': X1_train, 'X2': X2_train, 'Y': Y_train})
    sio.savemat('./../data_files/real_world_data/HCI_continuous/hci_cp_pairs_test_01.mat',
                {'X1': X1_test, 'X2': X2_test, 'Y': Y_test})



def augment_change_sequence_from_short_cp(X,cp_indices,buff, pair_length,rep_times):
    'Function to augment sequence to get similar arrays '
    cp_indices = cp_indices.reshape(-1)
    cp_indices = np.insert(cp_indices, 0, 200)
    X_sim = np.array([])
    X1_dissim = np.array([])
    X2_dissim = np.array([])
    X1_sim = np.array([])
    X2_sim = np.array([])
    X_sim_label = []
    X_segments = np.array([])
    sim_label = 0
    min_seg_length = 350
    for i in range(0,len(cp_indices) -2):
        temp = X[cp_indices[i]   : cp_indices[i+1]  , : ]
        temp2 = X[cp_indices[i+1] : cp_indices[i + 2] , : ]





        if (len(temp)) >  min_seg_length:
            if len(temp) > 2*pair_length:
                'Getting similar pairs if segment long enough'
                X_sim = np.array([])
                for t in range(0, math.floor(len(temp) / pair_length)):
                    temp_t = np.expand_dims(temp[t * pair_length: (t + 1) * pair_length, :], axis=0)
                    temp_tiled = np.tile(temp_t, (rep_times, 1))
                    X_segments = np.concatenate((X_segments, temp_tiled ), axis=0) if X_segments.size \
                        else temp_tiled
                    X_sim = np.concatenate((X_sim, temp_t), axis=0) if X_sim.size \
                        else temp_t #For getting similar pairs
                temp = temp_t.squeeze(axis = 0)  #For dissimilar pair if needed later
                X_sim_label.append(sim_label)
                sim_label += 1
                X1_temp = X_sim.repeat(X_sim.shape[0], axis=0)
                X2_temp = np.tile(X_sim, [X_sim.shape[0], 1, 1])  # Need to look at this

                X1_sim = np.concatenate((X1_sim, X1_temp), axis=0) if X1_sim.size \
                    else X1_temp
                X2_sim = np.concatenate((X2_sim, X2_temp), axis=0) if X2_sim.size \
                    else X2_temp

            elif len(temp) > min_seg_length:
                'Getting segment of fixed size'
                temp = X[cp_indices[i] - buff : cp_indices[i + 1]+buff, :]
                length_trunc = (len(temp) - pair_length)/2
                temp = np.delete(temp,np.arange(0,math.floor(length_trunc)),0)
                temp = np.delete(temp, np.arange(len(temp)-round(length_trunc), len(temp)),0)

                if len(temp) < 600:
                    temp = temp.append(temp,temp[-1,:], axis = 0)
                elif len(temp) > 600:
                    temp = temp[:-1,:]
                'Repeat segment a fixed number of times and store. Later used to generate dissimilar pairs'
                temp_tiled = np.tile(temp, (rep_times, 1))
                X_segments = np.concatenate((X_segments, np.expand_dims(temp_tiled, axis = 0) ), axis=0) if X_segments.size \
                    else np.expand_dims(temp_tiled , axis= 0)


            if len(temp2) > min_seg_length:
                temp2 = X[cp_indices[i+1] - buff: cp_indices[i + 2] + buff, :]
                length_trunc = (len(temp2) - pair_length) / 2
                temp2 = np.delete(temp2, np.arange(0, math.floor(length_trunc)), 0)
                temp2 = np.delete(temp2, np.arange(len(temp2) - round(length_trunc), len(temp2)), 0)

                if len(temp2) < 600:
                    temp2 = temp2.append(temp, temp[-1, :], axis=0)
                elif len(temp2) > 600:
                    temp2 = temp2[:-1, :]
                'Getting dissimilar pairs'
                X1_temp_dis = np.expand_dims(temp, axis =  0)
                X2_temp_dis = np.expand_dims(temp2,axis = 0)
                X1_dissim = np.concatenate((X1_dissim, X1_temp_dis), axis=0) if X1_dissim.size \
                    else X1_temp_dis
                X2_dissim = np.concatenate((X2_dissim, X2_temp_dis), axis=0) if X2_dissim.size \
                    else X2_temp_dis





    for i in range(0,2000):
        'get paris which are not adjacent'
        print('Augmenting pair no:{0}'.format(i))
        (p1,p2) = random.sample(range(1, len(X_segments)), 2)
        seg_1 = X_segments[p1]
        seg_2 = X_segments[p2]
        concatenated_signal = np.concatenate((seg_1,seg_2),axis = 0)
        index = len(seg_1)
        sim = check_cp_at_index(concatenated_signal,index,0.103,0.13)
        X1_temp = np.expand_dims(seg_1[0:pair_length,:], axis = 0 )
        X2_temp = np.expand_dims( seg_2[0:pair_length, :] , axis = 0)


        if sim ==0:
            'dissimilar'
            continue
        elif sim == -1:
            X1_dissim = np.concatenate((X1_dissim, X1_temp), axis=0) if X1_dissim.size \
                else X1_temp
            X2_dissim = np.concatenate((X2_dissim, X2_temp), axis=0) if X2_dissim.size \
                else X2_temp
        elif sim == 1:
            X1_sim = np.concatenate((X1_sim, X1_temp), axis=0) if X1_sim.size \
                else X1_temp
            X2_sim = np.concatenate((X2_sim, X2_temp), axis=0) if X2_sim.size \
                else X2_temp
            'similar and concatenate'

    X1 = np.concatenate((X1_sim, X1_dissim), axis=0)
    X2 = np.concatenate((X2_sim, X2_dissim), axis=0)
    Y1 = np.ones(X1_sim.shape[0])
    Y2 = -1 * np.ones(X1_dissim.shape[0])
    Y = np.concatenate((Y1, Y2), axis=0)
    rand_perm = random.sample(range(0, Y.shape[0]), Y.shape[0])
    Y = Y[rand_perm]
    X1 = X1[rand_perm, :, :]
    X2 = X2[rand_perm, :, :]
    train_ratio = 0.7

    X1_train = X1[0:int(train_ratio * len(X1)), :, :]
    X2_train = X2[0:int(train_ratio * len(X2)), :, :]
    Y_train = Y[0:int(train_ratio * len(Y))]

    X1_test = X1[int(train_ratio * len(X1)):, :, :]
    X2_test = X2[int(train_ratio * len(X2)):, :, :]
    Y_test = Y[int(train_ratio * len(Y)):]

    sio.savemat('./../data_files/real_world_data/HCI_continuous/hci_cp_pairs_600_train_01.mat',
                {'X1': X1_train, 'X2': X2_train, 'Y': Y_train})
    sio.savemat('./../data_files/real_world_data/HCI_continuous/hci_cp_pairs_600_test_01.mat',
                {'X1': X1_test, 'X2': X2_test, 'Y': Y_test})



def get_cp_pairs_from_segs(X, pair_length):
    X1_dissim = np.array([])
    X2_dissim = np.array([])
    X1_sim = np.array([])
    X2_sim = np.array([])
    X_segments = np.array([])

    for i in range(0, len(X) - 1):
        X1_temp = X[i]



        X1_temp = np.tile(X1_temp, (2, 1))


        temp_seg = np.expand_dims(X1_temp, axis=0)
        X_segments = np.concatenate((X_segments, temp_seg), axis=0) if X_segments.size \
            else temp_seg




    k = 0
    while len(X1_sim) < 200:
        'get paris which are not adjacent'

        print('Augmenting pair no:{0}'.format(k))
        (p1,p2) = random.sample(range(1, len(X_segments)), 2)
        seg_1 = X_segments[p1]
        seg_2 = X_segments[p2]
        concatenated_signal = np.concatenate((seg_1,seg_2),axis = 0)
        index = len(seg_1)
        #sim = check_cp_at_index(concatenated_signal,index,0.103,0.13)
        sim = check_cp_at_index(concatenated_signal, index, 0.18,0.22)
        X1_temp = np.expand_dims(seg_1[0:pair_length,:], axis = 0 )
        X2_temp = np.expand_dims( seg_2[0:pair_length, :] , axis = 0)


        if sim ==0:
            'dissimilar'
            continue
        elif sim == -1:
            X1_dissim = np.concatenate((X1_dissim, X1_temp), axis=0) if X1_dissim.size \
                else X1_temp
            X2_dissim = np.concatenate((X2_dissim, X2_temp), axis=0) if X2_dissim.size \
                else X2_temp
        elif sim == 1:
            X1_sim = np.concatenate((X1_sim, X1_temp), axis=0) if X1_sim.size \
                else X1_temp
            X2_sim = np.concatenate((X2_sim, X2_temp), axis=0) if X2_sim.size \
                else X2_temp
            'similar and concatenate'
        k = k +1

    X1 = np.concatenate((X1_sim, X1_dissim), axis=0)
    X2 = np.concatenate((X2_sim, X2_dissim), axis=0)
    Y1 = np.ones(X1_sim.shape[0])
    Y2 = -1 * np.ones(X1_dissim.shape[0])
    Y = np.concatenate((Y1, Y2), axis=0)
    rand_perm = random.sample(range(0, Y.shape[0]), Y.shape[0])
    Y = Y[rand_perm]
    X1 = X1[rand_perm, :, :]
    X2 = X2[rand_perm, :, :]
    train_ratio = 0.9

    X1_train = X1[0:int(train_ratio * len(X1)), :, :]
    X2_train = X2[0:int(train_ratio * len(X2)), :, :]
    Y_train = Y[0:int(train_ratio * len(Y))]

    X1_test = X1[int(train_ratio * len(X1)):, :, :]
    X2_test = X2[int(train_ratio * len(X2)):, :, :]
    Y_test = Y[int(train_ratio * len(Y)):]

    sio.savemat('./../data_files/real_world_data/HCI_continuous/hci_cp_pairs_freehand_600_train_01.mat',
                {'X1': X1_train, 'X2': X2_train, 'Y': Y_train})
    sio.savemat('./../data_files/real_world_data/HCI_continuous/hci_cp_pairs_freehand_600_test_01.mat',
                {'X1': X1_test, 'X2': X2_test, 'Y': Y_test})


x_range = 980000
x_range = 2000

get_cps_from_long_sequence = 1
if get_cps_from_long_sequence == 1:
    ''' Get changes poins from long sequence'''
    start_index = 1000

    #start_index = 500000
    x_range = len(feats)
    x_range = 20000
    '''Start index and range for robust mmd'''


    start_index = 411000
    x_range =  420000

    #start_index = 515900
    #x_range =  540000
    X = feats_s01[ start_index : x_range , :]
    labels = labels[start_index:x_range]
    window_size =  800


    #if mackey glass : CP thres - 0.15, HCI : window 800, 0.10 ; acti tracker, window :200

    window_size = 200
    #if HCI : 1
    threshold_cp = 0.025
    change_indices = gel_all_cps_in_sequence( X , window_size,  5, threshold_cp, labels)



aug_sequence = 0
if aug_sequence == 1:
    pair_length = 1000
    buff =800

    X_w_cp = X_w_cp
    #augment_change_sequence(X_w_cp,cp_indices,buff,pair_length,5)
    #augment_change_sequence_from_short_cp(X_w_cp, cp_indices, buff, pair_length, 3)
    get_cp_pairs_from_segs(X_segs,600)

pair_size = 300
buff_size  =20

X1,X2,Y = get_pairs_from_cp(feats_s01,change_indices,pair_size,buff_size)
sio.savemat('./../data_files/real_world_data/actitracker/pairs4mcp_really.mat',{'X1':X1,'X2':X2,'Y':Y})
print('Here')

