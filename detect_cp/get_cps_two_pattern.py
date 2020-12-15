import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import  random_split, DataLoader
from load_series import TimSeries_single_class
import  matplotlib.pyplot as plt
import mmd_util
from nn_models import  RNNSeriesEncoding
from optim import Optim
from scipy.signal import find_peaks
from pairs_from_cp import get_paris_from_cp, detect_cp_and_get_pairs
from scipy.signal import find_peaks

file_path = './../data_files/UCI_datasets/TwoPatterns/TwoPatterns_TRAIN.mat'

cuda = 1

if torch.cuda.is_available():
    if not cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

p_win = 20
f_win = 20
batch_size = 30
lr = 0.00005
series_data = TimSeries_single_class(file_path, p_win, f_win)




feats = series_data.return_series()

enc_dim = 10
data_dim = feats.shape[1]

'''
RNN_encoding = RNNSeriesEncoding(enc_dim, data_dim )


optimizer = Optim(RNN_encoding.parameters(), 'adam', lr= lr,
                   grad_clip= 0.1,
                   weight_decay= 0,
                   momentum=  0)
if cuda:
    RNN_encoding.cuda()

if cuda:
    cuda0 = torch.device('cuda:0')

mmd = np.zeros(len(feats))

sigma_list = mmd_util.median_heuristic(feats, beta= 1)
sigma_var = torch.FloatTensor(sigma_list).cuda()

no_epochs = 2
train_index = 4000

lam_0 = 0.3
lam_ae = 0.1
train_batches = DataLoader(series_data, batch_size= batch_size, shuffle = True)

g_clip = 0.25
train_loss = []


for ep in range(0,no_epochs):
    RNN_encoding.train()
    epoch_loss = []
    for i_batch, sample_batched in enumerate(train_batches):

        if i_batch > 2000:
            break
        for p in  RNN_encoding.rnn_enc_layer.parameters():
            p.data.clamp_(- 0.1, 0.1)
        for p in  RNN_encoding.rnn_dec_layer.parameters():
            p.data.clamp_(- 0.1, 0.1)

        RNN_encoding.zero_grad()
        print('Batch no {0}'.format(i_batch))
        X_p = sample_batched['feats_p'].float().cuda()
        X_f = sample_batched['feats_f'].float().cuda()
        X_p_enc, X_p_dec = RNN_encoding(X_p)
        X_p_enc_1 = X_p_enc[:, 0: int(p_win/2), :]
        X_p_enc_2 = X_p_enc[:, int(p_win /2) : int(p_win), :]
        X_f_enc, X_f_dec = RNN_encoding(X_f  )
        recon_loss =  torch.mean((X_f - X_f_dec) ** 2) + torch.mean((X_p - X_p_dec) ** 2)
        #
        optimizer.zero_grad()
        loss =  -(( mmd_util.batch_mmd2_loss(X_f_enc, X_p_enc, sigma_var) ).mean() - (mmd_util.batch_mmd2_loss(X_p_enc_1, X_p_enc_2, sigma_var)).mean() - lam_ae *recon_loss)
        print('loss')
        loss_dum = loss

        if np.isnan(loss_dum.cpu().detach().numpy()):
            print('Here')


        loss.backward()

        optimizer.step()
        epoch_loss.append(loss)
        print(loss)
    train_loss.append(sum(epoch_loss))
    #print('Loss:'+sum((epoch_loss)))

'''

'''
for i in range(0, train_index):
    feat_temp = feats[i - p_win: i + f_win, :]
    past = torch.from_numpy(feats[i - p_win: i, :]).cuda().float()
    past = past.unsqueeze(0)
    fut = torch.from_numpy(feats[i: i + f_win, :]).cuda().float()
    fut = fut.unsqueeze(0)
    labels = labels[i - p_win: i + f_win]

for i in range(train_index, len(feats)-f_win):
    feat_temp = feats[ i - p_win : i + f_win,:]
    past = torch.from_numpy(feats[i - p_win : i , :]).cuda().float()
    past = past.unsqueeze(0)
    fut = torch.from_numpy(feats[i : i + f_win, :]).cuda().float()
    fut = fut.unsqueeze(0)
    labels = labels[ i - p_win : i + f_win]


    mmd[i] = mmd_util.batch_mmd2_loss(past, fut, sigma_var).cpu().numpy()[0]
'''


'''

for i in range(train_index, len(feats)-f_win):
    RNN_encoding.eval()

    with torch.no_grad():
        feat_temp = feats[ i - p_win : i + f_win,:]
        past = torch.from_numpy(feats[i - p_win : i , :]).cuda().float()
        past = past.unsqueeze(0)
        fut = torch.from_numpy(feats[i : i + f_win, :]).cuda().float()
        fut = fut.unsqueeze(0)
        labels = labels[ i - p_win : i + f_win]
        Xp_enc,_ = RNN_encoding(past)
        Xf_enc, _ = RNN_encoding(fut)



    mmd[i] = mmd_util.batch_mmd2_loss(Xp_enc, Xf_enc, sigma_var).cpu().numpy()[0]
'''
if __name__ == "__main__":
    X1_sim_arr = np.array([])
    X2_sim_arr = np.array([])
    X1_dissim_arr = np.array([])
    X2_dissim_arr = np.array([])
    for i in range(0, feats.shape[0]):
        series = feats[i,:].reshape(-1,1)
        X1_sim, X2_sim, X1_dis, X2_dis = detect_cp_and_get_pairs(series,10)
        X1_sim_arr = np.concatenate((X1_sim_arr, X1_sim), axis=0) if X1_sim_arr.size else X1_sim
        X2_sim_arr = np.concatenate((X2_sim_arr, X2_sim),axis = 0) if X2_sim_arr.size else X2_sim
        X1_dissim_arr = np.concatenate((X1_dissim_arr, X1_dis), axis=0) if X1_dissim_arr.size else X1_dis
        X2_dissim_arr = np.concatenate((X2_dissim_arr , X2_dis), axis=0) if X2_dissim_arr.size else X2_dis
    print('done')

