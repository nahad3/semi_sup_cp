import os
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader



class CreateDataset4mDict(Dataset):
    'Class for returning Dataset from Dcitioary List. Used for balanced sample return'
    def __init__(self, list_dict):
        self.list_dict = list_dict

    def __len__(self):
        return len(self.list_dict)

    def  __getitem__(self, item):
        return self.list_dict[item]

class CreateDataset(Dataset):
    "Create data set from given X and Y"

    def __init__(self, X, Y):
        self.X= X
        self.Y = Y

    def __getitem__(self, idx):
        X = self.X[idx]
        Y = self.Y[idx]
        sample = {'X': X,  'Y':Y }
        return sample

    def __len__(self):
        return  len(self.Y)


class ConstraintDataset4mCP(Dataset):
    'C;'
    def __init__(self, sim_file, win_size):
        self.mat_file = sio.loadmat(sim_file)
        self.X1 = self.mat_file['X1']
        self.X2 = self.mat_file['X2']
        self.Y = self.mat_file['Y']

        dims = np.ndim(self.X1)

        'Data is assumed to be of 3 dimensions: no_segs x no_points_per_seg x dims series'
        'If it is not, make 3 dimensional'
        if dims < 3:
            self.X1 = np.expand_dims(self.X1, axis=2)
            self.X2 = np.expand_dims(self.X2, axis=2)

        'Increase labels to match no of points per segment'

        self.Y = np.multiply(self.Y.reshape(-1, 1), np.ones((self.X1.shape[0], self.X2.shape[1])))
        self.Y = self.Y.reshape(self.Y.shape[0], self.Y.shape[1], -1)
        if win_size is not None:
            self.X1 = self.get_windows(self.X1, win_size)
            self.X2 = self.get_windows(self.X2, win_size)
            self.Y = self.get_windows(self.Y, win_size)
        '''Get womdows non overlapping'''

    def get_windows(self, X, win_size):
        a_per_block = int(X.shape[1] / win_size)
        if a_per_block == 0:
            print("Window size too small")
            return -1
        else:
            temp = np.hsplit(X[:, 0:(a_per_block * win_size), :], a_per_block)
            return np.vstack(temp)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        X1 = self.X1[idx]
        X2 = self.X2[idx]
        Y = self.Y[idx]
        sample = {'X1': X1, 'X2': X2, 'Y': Y}
        return sample

class ConstrainDataset(Dataset):
    "Constrained Det Assumes 3 arrays for each pairL: X1(1st sample,X2(2nd Sample), Y(Similar/Dissimilar"
    def __init__(self, sim_file, win_size,from_cp = 0):
        self.mat_file = sio.loadmat(sim_file)
        self.X1 = np.array(self.mat_file['X1'])
        self.X2 = np.array(self.mat_file['X2'])
        self.Y = np.array(self.mat_file['Y'])


        dims = np.ndim(self.X1)

        'Data is assumed to be of 3 dimensions: no_segs x no_points_per_seg x dims series'
        'If it is not, make 3 dimensional'
        if dims < 3:
            self.X1 = np.expand_dims(self.X1, axis = 2)
            self.X2 = np.expand_dims(self.X2, axis = 2 )

        self.X1_label = np.array(self.mat_file['X1_label'])
        self.X2_label = np.array(self.mat_file['X2_label'])

        'Increase labels to match no of points per segment'
        self.X1_label = np.multiply(self.X1_label.reshape(-1,1), np.ones((self.X1.shape[0],self.X1.shape[1])))
        self.X2_label = np.multiply(self.X2_label.reshape(-1, 1), np.ones((self.X1.shape[0], self.X2.shape[1])))
        self.X1_label = self.X1_label.reshape(self.X1_label.shape[0], self.X1_label.shape[1],-1)
        self.X2_label = self.X2_label.reshape(self.X2_label.shape[0], self.X2_label.shape[1], -1)
        self.Y = np.multiply(self.Y.reshape(-1,1), np.ones((self.X1.shape[0],self.X2.shape[1])))
        self.Y = self.Y.reshape(self.Y.shape[0], self.Y.shape[1],-1)
        if win_size is not None:
            self.X1 = self.get_windows(self.X1, win_size)
            self.X2 = self.get_windows(self.X2,win_size)
            self.X1_label = self.get_windows(self.X1_label,win_size)
            self.X2_label = self.get_windows(self.X2_label, win_size)
            self.Y = self.get_windows(self.Y, win_size)

    def __len__(self):
        return  len(self.Y)

    def __getitem__(self, idx):
        X1 = self.X1[idx]
        X2 = self.X2[idx]
        X1_label = self.X1_label[idx]
        X2_label = self.X2_label[idx]
        Y = self.Y[idx]
        sample = {'X1': X1, 'X2': X2, 'Y':Y ,'X1_label':X1_label,'X2_label':X2_label}
        return sample

    '''Get womdows non overlapping'''
    def get_windows(self, X, win_size):
        a_per_block = int(X.shape[1]/win_size)
        if a_per_block == 0 :
            print("Window size too small")
            return -1
        else:
            temp = np.hsplit(X[:,0:(a_per_block * win_size),:], a_per_block)
            return np.vstack(temp)



class LabelledDataset(Dataset):
    'Class for loading Labelled Data which is continuous'
    def __init__(self, sim_file, win_size):
        self.mat_file = sio.loadmat(sim_file)

        'loading similar files'


        self.X_series = np.array(self.mat_file['X'])
        self.X = self.X_series
        dims = np.ndim(self.X)
        'Data is assumed to be of 3 dimensions: no_segs x no_points_per_seg x dims series'
        'If it is not, make 3 dimensional'
        if dims < 2:
            self.X = np.expand_dims(self.X, axis=1)
        self.Y = np.array(self.mat_file['Y'])
        self.Y = self.Y.reshape(-1,1)
        if win_size is not None:
            #self.X = get_slid_windws_feats(self.X, win_size)
            #self.Y = get_slid_windws_labels(self.Y, win_size)
            self.X = self.get_windows(self.X,win_size)
            self.Y = self.get_windows(self.Y,win_size)

    def __len__(self):
        return  len(self.Y)

    def get_series(self):
        X = self.X_series
        return  X

    def __getitem__(self, idx):
        X = self.X[idx]
        Y = self.Y[idx]
        sample = {'X': X,  'Y':Y }
        return sample

    def get_windows(self, X, win_size):
        a_per_block = int(X.shape[0] / win_size)
        if a_per_block == 0:
            print("Window size too small")
            return -1
        else:
            temp = np.vsplit(X[ 0:(a_per_block * win_size), :], a_per_block)
            return np.stack(temp)


class SegmentedLabelledDataset(Dataset):
    'Access labelled segments stored in the form of cells'
    def __init__(self, sim_file):
        self.mat_file = sio.loadmat(sim_file)
        self.X = self.mat_file['X'].reshape(-1)
        self.Y  = self.mat_file['Y'].reshape(-1)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        X = self.X[idx]
        Y = self.Y[idx]
        sample = {'X': X, 'Y': Y}
        return sample




class load_sim_data:

    def __init__(self, file_path, train_ratio):
        self.mat_file = sio.loadmat(file_path)
        'loading similar files'
        self.X1_sim = np.array(self.mat_file['X1_sim'])
        self.X2_sim = np.array(self.mat_file['X2_sim'])
        self.train_ratio = train_ratio
        self.X1_dis = np.array(self.mat_file['X1_dissim'])
        self.X2_dis = np.array(self.mat_file['X2_dissim'])
        self.no_sim_points = self.X1_sim.shape[0]
        self.no_dissim_points = self.X1_dis.shape[0]
        self.no_train_sim = int(self.no_sim_points * self.train_ratio)
        # validation test split (add to 1
        self.val_test_split = [0.4, 0.6]

    def get_train(self, batch_size):

        X1_train = self.X1_sim[0 : self.no_train_sim , :, :]
        X2_train = self.X2_sim[0 : self.no_train_sim , :, :]

        X1_train = np.expand_dims(X1_train,axis = 0)
        X2_train = np.expand_dims(X2_train,axis = 0)
        Y_sim  = np.ones(self.no_train_sim).reshape(1,-1, 1 , 1)
        no_batches = int( self.no_train_sim / float(batch_size))
        X1_sim_batch = [X1_train[i: i + batch_size] for i in range(0, no_batches)]
        X2_sim_batch = [X2_train[i: i + batch_size] for i in range(0, no_batches)]
        Y_sim = [np.ones(batch_size) for i in range(0,no_batches)]


        numb_points= self.X1_dis.shape[0]
        X1_train = self.X1_dis[0 : int( numb_points * self.train_ratio ), :, :]
        X2_train = self.X2_dis[0 : int(numb_points * self.train_ratio ), :, :]
        no_batches = int((numb_points * self.train_ratio) / float(batch_size))
        no_batches = max(1, no_batches)
        X1_dissim_batch = [X1_train[i: i + batch_size] for i in range(0, no_batches)]
        X2_dissim_batch = [X2_train[i: i + batch_size] for i in range(0, no_batches)]
        Y_dissim = [-1*np.ones(batch_size) for i in range(0, no_batches)]

        train_batches = [[X1_sim_batch, X2_sim_batch, Y_sim], [X1_dissim_batch, X2_dissim_batch, Y_dissim]]
        return

    def get_val_and_test(self, batch_size):
        #obtaining sets for similar segments

        numb_points= self.X1_sim.shape[0]
        no_validation = numb_points - int((1 - self.train_ratio))
        X1_val = self.X1_sim[int(numb_points * self.train_ratio):
                             int(numb_points * self.train_ratio + no_validation * self.val_test_split[0]), :]
        X2_val = self.X2_sim[int(numb_points * self.train_ratio):
                             int(numb_points * self.train_ratio + no_validation * self.val_test_split[0]), :]

        X1_test = self.X1_sim[int(numb_points * self.train_ratio + no_validation * self.val_test_split[0]) : , :]
        X2_test = self.X2_sim[int(numb_points * self.train_ratio + no_validation * self.val_test_split[0]) :, :]

        no_batches_val = int((no_validation * self.val_test_split[0]) / float(batch_size))
        no_batches_val = max(1, no_batches_val)
        no_batches_test = int((no_validation * self.val_test_split[1]) / float(batch_size))
        no_batches_test = max(1, no_batches_test)

        X1_sim_val_batch = [X1_val[i: i + batch_size] for i in range(0, no_batches_val)]
        X2_sim_val_batch = [X2_val[i: i + batch_size] for i in range(0, no_batches_val)]

        X1_sim_test_batch = [X1_test[i: i + batch_size] for i in range(0, no_batches_test)]
        X2_sim_test_batch = [X2_test[i: i + batch_size] for i in range(0, no_batches_test)]

        X_sim_val = (X1_sim_val_batch, X2_sim_val_batch)
        X_sim_test = (X1_sim_test_batch, X2_sim_test_batch)

       #obtaining sets for dissimilar segments

        numb_points= self.X1_dis.shape[0]
        no_validation = numb_points - int((1 - self.train_ratio))
        X1_val = self.X1_dis[int(numb_points * self.train_ratio):
                             int(numb_points * self.train_ratio + no_validation * self.val_test_split[0]), :]
        X2_val = self.X2_dis[int(numb_points * self.train_ratio) :
                             int(numb_points * self.train_ratio + no_validation * self.val_test_split[0]), :]

        X1_test = self.X1_dis[int(numb_points * self.train_ratio + no_validation * self.val_test_split[0]) :, :]
        X2_test = self.X2_dis[int(numb_points * self.train_ratio + no_validation * self.val_test_split[0]) :, :]

        no_batches_val = int((no_validation * self.val_test_split[0]) / float(batch_size))
        no_batches_val = max(1, no_batches_val)
        no_batches_test = int((no_validation * self.val_test_split[1]) / float(batch_size))
        no_batches_test = max(1, no_batches_test)
        X1_dis_val_batch = [X1_val[i: i + batch_size] for i in range(0, no_batches_val)]
        X2_dis_val_batch = [X2_val[i: i + batch_size] for i in range(0, no_batches_val)]

        X1_dis_test_batch = [X1_test[i: i + batch_size] for i in range(0, no_batches_test)]
        X2_dis_test_batch = [X2_test[i: i + batch_size] for i in range(0, no_batches_test)]

        X_dis_val = (X1_dis_val_batch, X2_dis_val_batch)
        X_dis_test = (X1_dis_test_batch, X2_dis_test_batch)

        return (X_sim_val, X_dis_val), (X_sim_test, X_dis_test)


#Class load dissim two pattern dataset
class sim_disim_unqual_length(Dataset):
    def __init__(self, file_path):
        self.mat_file = sio.loadmat(file_path)
        self.X1_sim = self.mat_file['X1_sim']
        self.X2_sim = self.mat_file['X2_sim']
        self.X1_dis = self.mat_file['X1_dis']
        self.X2_dis = self.mat_file['X2_dis']

    def __len__(self):
        return  self.X1_sim.shape[0]+self.X1_dis.shape[0]

    def __getitem__(self, item):
        if item < self.X1_sim.shape[0]:
            X1 = self.X1_sim[item,:,:]
            X2 = self.X2_sim[item,:,:]
            Y = 1
        else:
            item = item - self.X1_sim.shape[0]
            X1 = self.X1_dis[item ,:,:]
            X2 = self.X2_dis[item ,:,:]
            Y = -1
        sample = { 'X1' : X1, 'X2':X2 , 'Y': Y}
        return  sample


def get_uniform_samples(ds, no_classes, samp_class):
    sampled_data = []
    for i in range(0,no_classes):
        count = 0
        while(count < samp_class):
            for j in range(len(ds)):
                label = ds[j]['Y'].mean()
                if label == i:
                    count += 1
                    sampled_data.append(ds[j])
                    if count == samp_class:
                        break


    return sampled_data



'Get sliding windows for features'
def get_slid_windws_feats(X,win_size):
    x_temp = list()

    for i in range(win_size, X.shape[0]):
        x_temp.append(X[i - win_size:i, :])
        # x_temp = np.append((x_temp, X[i-win_size:i,:]) ,axis = 0) if x_temp.size else X[i-win_size:i,:]
    return  np.array(x_temp)



'Get s;odomg womdpws for Labels'
def get_slid_windws_labels(Y, win_size):
    y_temp =list()

    for i in range(win_size, Y.shape[0]):
        buffer = Y[i - win_size:i]
        y_temp.append(buffer)

    y_temp = np.array(y_temp)
    return  y_temp
