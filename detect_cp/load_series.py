import os
import numpy as np
import scipy.io as sio
from random import shuffle
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import scale


class Get_seg_data(Dataset):
    def __init__(self, file_path):
        self.mat_file = sio.loadmat(file_path)
        self.X = self.mat_file['X'].reshape(-1)
    def return_segs(self):
        return  self.X

class SeriesDataset(Dataset):
    def __init__(self, file_path, p_wind, f_wind):
        self.mat_file = sio.loadmat(file_path)
        self.feats = self.mat_file['X']
        self.feats = np.nan_to_num(self.feats)
        self.feats_0_1 = self.feats
        self.cp_ind_available = 0
        if "cp_indices" in self.mat_file:
            self.change_indices = self.mat_file['cp_indices']
            self.cp_ind_available = 1
        self.feats = scale(self.feats)
        self.labels = np.asarray(self.mat_file['Y']).reshape(-1,)
        self.changes = np.diff(self.labels)
        self.changes = np.not_equal(self.changes,0).astype(int)
        self.changes = np.insert(self.changes, 0, 0)
        self.p_wind = p_wind
        self.f_wind = f_wind
        a, b = np.asarray(self.feats).shape
        self.length = a
        self.dims = b

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        'For reuring past and future windows incase need to get cps in an encoded space'
        if item - self.p_wind < 0:
            item = self.p_wind
        if item + self.f_wind > self.length - 1:
            item = self.length - 1 - self.f_wind

        feats = self.feats[item - self.p_wind: item + self.f_wind, :]
        labels = self.labels[item - self.p_wind : item + self.f_wind]
        change = self.changes[item]
        feats_p = self.feats[item - self.p_wind : item, :]
        feats_f = self.feats[item: item + self.f_wind, :]
        sample = {'feat': feats, 'label': labels, 'change': change, 'feats_p':feats_p, 'feats_f': feats_f}
        return sample

    def return_series(self):
        feats = self.feats
        labels = self.labels
        feats_s_0_1 = self.feats_0_1
        if self.cp_ind_available == 0:
            return feats , feats_s_0_1,labels
        else:
            return feats, feats_s_0_1, labels, self.change_indices

class Read_signal_and_cp(Dataset):
    def __init__(self, file_path):
        self.mat_file = sio.loadmat(file_path)
        self.X =   self.mat_file['X']
        self.cp_indices = self.mat_file['c_indices']

    def read_series(self):
        return  self.X,self.cp_indices


class TimSeries_single_class(Dataset):
    def __init__(self, file_path, p_wind, f_wind):
        self.mat_file = sio.loadmat(file_path)
        self.feats = self.mat_file['X']
        self.feats = np.nan_to_num(self.feats)
        self.feats = scale(self.feats, axis = 1)
        self.p_wind = p_wind
        self.f_wind = f_wind
        a, b = np.asarray(self.feats).shape
        self.length = a
        self.dims = b

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        if item - self.p_wind < 0:
            item = self.p_wind
        if item + self.f_wind > self.length - 1:
            item = self.length - 1 - self.f_wind

        feats = self.feats[item - self.p_wind: item + self.f_wind]
        feats_p = self.feats[item - self.p_wind : item]
        feats_f = self.feats[item: item + self.f_wind, :]
        sample = {'feat': feats, 'feats_p':feats_p, 'feats_f': feats_f}
        return sample

    def return_series(self):
        feats = self.feats
        return feats





