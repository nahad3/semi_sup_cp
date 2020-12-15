import numpy as np
import mmd_util
import torch
from scipy.signal import find_peaks


'''get_pairs_from_cp takes in a signal alond with identified cps. These cps are then used \
to obtain similar and dissimilar pairs'''

def get_pairs_from_cp(signal, cp_indices,seg_size,buff):
    #seg_size has to be even
    cp_indices = cp_indices.tolist()
    sig_dim = signal.shape[-1]
    X1= np.array([])
    X2= np.array([])


    sim_size = seg_size if seg_size % 2 == 0 else seg_size-1

    Y = np.array([])
    if signal.shape[0] - cp_indices[-1] - buff < 2 * seg_size:
        cp_indices = cp_indices[:-1]

    for cp in cp_indices:
        'Checking if sufficient length available for last cp to get segment after cp'

        Xp_1 = signal[cp - 2 * seg_size - buff: cp - seg_size - buff, :]
        Xp_2 = signal[cp - seg_size - buff: cp - buff, :]
        Xf_1 = signal[cp + buff: cp + buff + seg_size,:]
        Xf_2 = signal[cp + buff + seg_size: cp + buff + 2 * seg_size, :]
        Xp_1 = Xp_1.reshape(1, seg_size, -1)
        Xp_2 = Xp_2.reshape(1, seg_size, -1)
        Xf_1 = Xf_1.reshape(1, seg_size, -1)
        Xf_2 = Xf_2.reshape(1, seg_size, -1)

        Y = np.concatenate((Y, np.asarray([-1, -1,1,1])), axis=0) if Y.size \
            else np.asarray([-1,-1,1,1])
        X1_temp = np.concatenate((Xp_1,Xp_2,Xp_1,Xf_1),axis = 0)
        X2_temp = np.concatenate((Xf_1,Xf_2,Xp_2,Xf_2),axis = 0)
        X1 = np.concatenate((X1,X1_temp),axis = 0) if X1.size \
            else X1_temp
        X2 = np.concatenate((X2, X2_temp), axis=0) if X2.size \
            else X2_temp
    return X1,X2,Y



'''detect_cp_and_get_pairs applies change points on a signal using the MMD statistic and find cp indices \
Once the points, are identified, it calls the get_pairs_from_cp function to get similar and dissimilar pairs'''


def detect_cp_and_get_pairs(signal, win_size):
    #Reshape signal such that first dimension consists of number of points. 2nd dimension: signal dimension
    series = signal.reshape(-1, 1)
    mmd = np.zeros(series.shape[0])
    sigma_list = mmd_util.median_heuristic(series, beta=0.1)
    sigma_var = torch.FloatTensor(sigma_list).cuda()
    p_win = win_size
    f_win = win_size
    feat_temp = series
    for i in range(win_size, series.shape[0] - f_win):
        past = torch.from_numpy(feat_temp[i - p_win: i, :]).cuda().float()
        past = past.unsqueeze(0)
        fut = torch.from_numpy(feat_temp[i: i + f_win, :]).cuda().float()
        fut = fut.unsqueeze(0)
        mmd[i] = mmd_util.batch_mmd2_loss(past, fut, sigma_var).cpu().numpy()[0]
    change_indices = find_peaks(mmd, height=1.0)[0]
    X1_sim, X2_sim, X1_dis, X2_dis = get_paris_from_cp(series, change_indices, 5)
    return X1_sim, X2_sim, X1_dis, X2_dis