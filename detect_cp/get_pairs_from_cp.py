import torch
import numpy as np
from scipy.signal import find_peaks
import detect_cp.mmd_util as mmd_util

'Contains Class to get segment pairs from signal from change points'
"Usage requires passing a signal"


class Get_pairs_from_cp():
    def __init__(self,signal,threshold,seg_win,pair_length,buffer):
        '''signal: Signal on which CP to be detected
        threshold: Threshold to detect CP
        seg_win : segment window for mmd to detect cp (rolling windows of future and past windows of size seg_win
        pair_length: length of each segment in similar/dissimilar pairs returned
        buffer: buffer before and after c efore obtaining pair'''
        self.signal = signal
        self.threshold = threshold
        self.seg_win = seg_win
        self.pair_length = pair_length
        self.buffer = buffer



    def get_cp_indices(self):
        mmd = np.zeros(len(self.signal))
        sigma_list = mmd_util.median_heuristic(self.signal, beta=0.1)
        sigma_var = torch.FloatTensor(sigma_list).cuda()
        for i in range(2*self.pair_length, self.signal.shape[0] - self.pair_length):
            past_np = self.signal[i - self.pair_length: i, :]
            past = torch.from_numpy(past_np).cuda().float()
            past = past.unsqueeze(0)
            fut = torch.from_numpy(self.signal[i: i + self.pair_length, :]).cuda().float()
            fut = fut.unsqueeze(0)
            # labels = labels[ i - p_win : i + f_win]
            sigma_list = mmd_util.median_heuristic(past_np, beta=1)
            sigma_var = torch.FloatTensor(sigma_list).cuda()
            mmd[i] = mmd_util.batch_mmd2_loss(past, fut, sigma_var).cpu().numpy()[0]
        self.change_indices = find_peaks(mmd, distance=3*self.seg_win, height=self.threshold)[0]


    def get_segments(self):
        signal = self.signal
        buff = self.buffer
        self.get_cp_indices()
        cp_indices = self.change_indices.tolist()
        seg_size = self.pair_length
        #seg_size has to be even

        sig_dim = signal.shape[-1]
        X1= np.array([])
        X2= np.array([])

        seg_size = seg_size if seg_size % 2 == 0 else seg_size-1

        Y = np.array([])
        if signal.shape[0] - cp_indices[-1] - buff < 2 * seg_size :
            cp_indices = cp_indices[:-1]
        for cp in cp_indices:
            'Checking if sufficient length available for last cp to get segment after cp'


            Xp_1 = signal[cp - 2*seg_size - buff : cp - seg_size - buff, :]
            Xp_2 = signal[cp - seg_size - buff: cp - buff ,  :]
            Xf_1 = signal[cp + buff : cp + buff +seg_size :]
            Xf_2 = signal[cp + buff + seg_size: cp + buff+ 2*seg_size, :]
            Xp_1 = Xp_1.reshape(1,seg_size ,-1)
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