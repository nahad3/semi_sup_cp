import torch
import sklearn
import numpy as np
from models import TCN, NN_classification
from models import Autoencoder
from data_loader import load_sim_data
from data_loader import ConstrainDataset
from data_loader import LabelledDataset, get_uniform_samples, CreateDataset, CreateDataset4mDict, ConstraintDataset4mCP, \
    SegmentedLabelledDataset
from torch.utils.data import random_split, DataLoader
from distance import KLDiv
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os
from torch import nn
from ladder import Ladder as LadderNetwork
from utils import increase_all_pairs, PairEnum, Class2Simi
import matplotlib.pyplot as plt
from detect_cp.get_pairs_from_cp import Get_pairs_from_cp
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.manifold import TSNE
from sklearn.semi_supervised import LabelPropagation
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import torch.nn.functional as F
import random
from scipy.stats import mode
from utils import get_f1_score
from torch.autograd import Variable
from torch.distributions import kl
import sys


'This script is for the HCI example only'
labels = 0
auto_enc = 1

torch.cuda.manual_seed(5)
# torch.cuda.manual_seed_all(5)  # if you are using multi-GPU.
np.random.seed(5)  # Numpy module.
random.seed(5)  # Python random module.
torch.manual_seed(5)
# torch.backends.cudnn.benchmark = False

# torch.backends.cudnn.deterministic = True

temp = 1
pair_sig_cost = 1
full_sup = 0

mean_emb = 1

'''Tensor board writing path'''
os.system('rm -rf /home/nahad/pycharm_mac/semi_with_cp/logs/*')
writer_path = '/home/nahad/pycharm_mac/semi_with_cp/logs'
writer = SummaryWriter(writer_path)

train_samples = 6

semi_sup = 1
ae = 1
ladder_var = 0
paris_from_cp = 0

# Select case
case = 6

save_iters = 0

# case 1: AR var/mean change
# case 2: AR process (coeff change)
# case 3: Mackay switch
# case 4 : ActiTracker


model_save_base_path = '/home/nahad/pycharm_mac/semi_with_cp/semi_ts_clustering/models'
if paris_from_cp == 1:
    file_path_pairwise = './data_files/human_activity/mat_files/HARpair4mcp_HighThresh.mat'
else:
    file_path_pairwise = './data_files/human_activity/mat_files/Train_scaled_0_1_sim_dissim_from_cp_with_labels_ActivTraker2011_.mat'
    file_path_pairwise = './data_files/gen_simulations/mat_files/ar_process/ar_train_pairs.mat'

levels = 3
n_channels = [100, 100, 100]


batch_size_label = 300
batch_size_pr = 300
train_ratio_pair = 0.30
train_ratio_label = 0.50  # 60 labels (when 6 labels used)
train_ratio_label = 0.90
train_ratio_pair = 0.70
# For acti tracker this has to be maintained
# train_ratio_label = 0.30

labels_segmented = 0

hingeloss1 = nn.HingeEmbeddingLoss(margin=8.0, reduction='sum')
hingeloss2 = nn.HingeEmbeddingLoss(margin=8.0, reduction='mean')

if case == 1:
    file_path_pairwise = './data_files/gen_simulations/mat_files/mean_var_change/mean_var_chg_train_pairs.mat'
    file_path_labels = './data_files/gen_simulations/mat_files/mean_var_change/mean_var_chg_train_labels.mat'
    test_label_path = './data_files/gen_simulations/mat_files/mean_var_change/mean_var_chg_test_labels.mat'
    test_pairs_path = './data_files/gen_simulations/mat_files/mean_var_change/mean_var_chg_train_pairs.mat'
    output_size = 5
    input_size = 1
    kernel_size = 2
    win_size = 10
elif case == 2:
    file_path_pairwise = './data_files/gen_simulations/mat_files/ar_coeffs_change/ar_chg_train_pairs.mat'
    file_path_labels = './data_files/gen_simulations/mat_files/ar_coeffs_change/ar_signal_train.mat'
    test_label_path = './data_files/gen_simulations/mat_files/ar_coeffs_change/ar_signal_test.mat'
    test_pairs_path = './data_files/gen_simulations/mat_files/ar_coeffs_change/ar_chg_test_pairs.mat'
    output_size = 5
    input_size = 1
    kernel_size = 30
    win_size = 200
elif case == 3:
    file_path_pairwise = './data_files/gen_simulations/mat_files/mackay_switch/switch_mackay_pair_train.mat'
    file_path_labels = './data_files/gen_simulations/mat_files/mackay_switch/switch_mackay_series_train.mat'
    test_label_path = './data_files/gen_simulations/mat_files/mackay_switch/switch_mackay_series_test.mat'
    test_pairs_path = './data_files/gen_simulations/mat_files/mackay_switch/switch_mackay_pair_test.mat'
    output_size = 4
    input_size = 1
    kernel_size = 5
    win_size = 200

elif case == 4:
    levels = 4
    # n_channels = [100, 100, 100,100]
    n_channels = [100, 100, 100]
    file_path_pairwise = './data_files/gen_simulations/mat_files/mackay_switch/noisy_switch_mackay_pair_train.mat'
    file_path_labels = './data_files/gen_simulations/mat_files/mackay_switch/noisy_switch_mackay_series_train.mat'
    test_label_path = './data_files/gen_simulations/mat_files/mackay_switch/noisy_switch_mackay_series_test.mat'
    test_pairs_path = './data_files/gen_simulations/mat_files/mackay_switch/noisy_switch_mackay_pair_test.mat'
    output_size = 4
    input_size = 1
    kernel_size = 10
    win_size = 100

elif case == 5:
    'pairs from true changes'
    # file_path_pairwise = './data_files/human_activity/mat_files/Train_scaled_0_1_sim_dissim_from_cp_with_labels_ActivTraker2011_.mat'

    'from change points'
    # file_path_pairwise =  './data_files/real_world_data/actitracker/pairs4mactualCP.mat'

    'from change points split and cleaned '
    file_path_pairwise = './data_files/real_world_data/actitracker/pairs4mactualCP_refined.mat'

    file_path_labels = './data_files/human_activity/mat_files/Train_scaled_0_1_ActivTraker2011.mat'
    test_label_path = './data_files/human_activity/mat_files/Test_scaled_0_1_ActivTraker2011.mat'
    test_pairs_path = './data_files/human_activity/mat_files/Test_scaled_0_1_sim_dissim_from_cp_with_labels_ActivTraker2011_.mat'
    output_size = 6
    input_size = 3
    kernel_size = 10
    win_size = 50
    paris_from_cp = 1
    train_ratio_label = 0.30
    train_ratio_pair = 0.30
elif case == 6:
    # file_path_pairwise = './data_files/real_world_data/HCI_continuous/hci_cp_pairs_600_train_01.mat'
    file_path_pairwise = './data_files/real_world_data/HCI_continuous/hci_cp_pairs_freehand_600_train_01.mat'
    # file_path_labels = './data_files/real_world_data/HCI_continuous/hci_cp_untrunc_guided_raw_rescaled01.mat'
    # file_path_labels = './data_files/real_world_data/HCI_continuous/labelled_segmented_freehand_01_HCI.mat'
    file_path_labels = './data_files/real_world_data/HCI_continuous/labelled_segmented_freehand_01_train_HCI.mat'
    test_label_path = './data_files/real_world_data/HCI_continuous/labelled_segmented_freehand_01_test_HCI.mat'
    # file_path_labels = './data_files/real_world_data/HCI_continuous/labelled_segmented_guided_01_HCI.mat'
    test_pairs_path = './data_files/real_world_data/HCI_continuous/hci_cp_pairs_600_test_01.mat'
    output_size = 6
    input_size = 48
    kernel_size = 30
    win_size = 600
    paris_from_cp = 1
    labels_segmented = 1
    hingeloss1 = nn.HingeEmbeddingLoss(margin=4.0, reduction='sum')
    hingeloss2 = nn.HingeEmbeddingLoss(margin=4.0, reduction='mean')
    train_ratio_pair = 0.30
    n_channels = [100, 100, 100]

'''Setting model Parameters'''
# Number of dimensions of input (channels)

# Number of dimensions of output (Should be the cluster)
output_size_tcn = output_size

# Filter size


# Like hidden unites LST<s
# n_channels =[100,100,100]

dropout = 0.0

lr = 0.05
cuda = 1
epochs_labels = 1200
epochs_pairs = 120
g_clip = 0

# Dataset parameters


'For AR tests'
# train_ratio_label = 0.00138 # 40 labels


'For HAR'

# train_ratio_label = 0.002


'''Load Data''' ''
if paris_from_cp == 1:
    pair_data_train = ConstraintDataset4mCP(file_path_pairwise, win_size)
    pair_data_test = ConstraintDataset4mCP(test_pairs_path, win_size)
else:
    pair_data_train = ConstrainDataset(file_path_pairwise, win_size)
    pair_data_test = ConstrainDataset(test_pairs_path, win_size)

# win_size = 200

if labels_segmented == 1:
    label_data_train = SegmentedLabelledDataset(file_path_labels)
    label_data_test = SegmentedLabelledDataset(test_label_path)
else:
    label_data_train = LabelledDataset(file_path_labels, win_size)

    label_data_test = LabelledDataset(test_label_path, win_size)

'''
data_series = label_data_train.get_series()


data_series_temp = data_series[381200:391200]
train_cp = Get_pairs_from_cp(data_series_temp,0.1,200,200,200)
X1,X2,Y = train_cp.get_segments()
dict_pairs = {'X1':X1,'X2':X2,'Y':Y}

ConstraintDataset4mCP(dict_pairs,win_size,from_cp=1)


'''

train_set_lab, val_set_lab = random_split(label_data_train, [int(train_ratio_label * len(label_data_train)), \
                                                             len(label_data_train) - int(train_ratio_label * \
                                                                                         len(label_data_train))])

print('Number of Train Labels:{0}'.format(train_samples))

train_set_pair, val_set_pair = random_split(pair_data_train, [int(train_ratio_pair * len(pair_data_train)), \
                                                              len(pair_data_train) - int(
                                                                  train_ratio_pair * len(pair_data_train))])

if case == 6:
    ul_split = 0.9
else:
    ul_split = 0.5

unlabelled_data, val_set_lab = random_split(val_set_lab, [int(ul_split * len(val_set_lab)),
                                                          len(val_set_lab) - int(ul_split * len(val_set_lab))])
unlabelled_data = train_set_lab

label_train_btch = DataLoader(train_set_lab, batch_size=batch_size_label, shuffle=True)
label_val_btch = DataLoader(val_set_lab, batch_size=batch_size_label, shuffle=True)

pair_train_btch = DataLoader(train_set_pair, batch_size=batch_size_pr, shuffle=False)
pair_val_btch = DataLoader(val_set_pair, batch_size=batch_size_pr, shuffle=False)

label_data_test = DataLoader(label_data_test, batch_size=len(label_data_test), shuffle=True)
pair_data_test = DataLoader(pair_data_test, batch_size=batch_size_pr)

ul_data_batch = DataLoader(unlabelled_data, batch_size=batch_size_label)

'Choosew method type to Define Models and set up optimizers'

if ladder_var == 1:
    model_LadderNet = LadderNetwork(input_size, output_size_tcn, kernel_size)
    model_label = NN_classification(output_size_tcn, output_size)
    temp = 1
    if cuda:
        model_LadderNet.cuda()
        model_label.cuda()
        cuda0 = torch.device('cuda:0')
    optimizer_ladder = optim.Adam(list(model_LadderNet.parameters()) + list(model_label.parameters()), lr=0.01)
if ae == 1:
    temp_ae = 1
    modelAE = Autoencoder(input_size, output_size_tcn, n_channels, kernel_size, dropout=dropout)
    model_label = NN_classification(output_size_tcn, output_size)
    if cuda:
        model_label.cuda()
        modelAE.cuda()
        cuda0 = torch.device('cuda:0')
    optimizer_ae = optim.Adam(list(modelAE.parameters()) + list(model_label.parameters()), lr=0.0001)

if semi_sup == 1:
    temp_semi = 10  # Mackay
    if case == 6 or case == 1:
        temp_semi = 5  # Mean var

    model1 = TCN(input_size, output_size_tcn, n_channels, kernel_size, temp_semi, dropout, win_size, 1)
    model = nn.Sequential(model1, nn.Softmax(dim=1))
    #model_label = NN_classification(output_size_tcn, output_size)
    if cuda:
        model.cuda()
        #model_label.cuda()
        cuda0 = torch.device('cuda:0')
    'Define Optimizer'
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer_classifier = optim.Adam(model_label.parameters(), lr=0.0001)

if torch.cuda.is_available():
    if not cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

'Setting losses'

criterion_CrsEnt = nn.CrossEntropyLoss()
criterion_MSE = nn.MSELoss()
criterion_CosSim = nn.CosineEmbeddingLoss(margin=1.0, reduction='sum')

'Use for PI semi-sup'
kl_div1 = nn.KLDivLoss(reduction='mean')
kl_div2 = nn.KLDivLoss(reduction='mean')
'Move model to GPU'


def get_balanced_samples(train_data, samples, classes):
    'Function for obtaining balanced set of samples'
    different_seed_label = []
    for i in range(0,100):

        s_per_class = int(samples / classes)
        data_list = list(train_data)
        label_list = np.array([int((np.round(mode(x['Y']).mode[0]))) for x in data_list])
        list_balanced = []
        for i in range(0, classes):
            idx = np.where(label_list == i)[0].tolist()
            random.shuffle(idx)
            idx = idx[0:s_per_class]
            list_balanced.extend([data_list[k] for k in idx])
        random.shuffle(list_balanced)
        balanced_train = CreateDataset4mDict(list_balanced)
        different_seed_label.append(balanced_train)
    return different_seed_label


def get_input_prob(x):
    x = torch.tensor([x, x]).unsqueeze(0).unsqueeze(2)
    with torch.no_grad():
        out = model(x.cuda().float())
        return out


'Code for training labels'


def train_labels(data_batched, ep):
    model_label.train()
    # model1.train()
    model1.eval()
    # model1.train()
    # model1.requires_grad = False

    # for param in model_label.parameters():
    #   param.requires_grad = True

    loss_list = []
    'Load Batch and optimize for eac  h batch'
    for i_batch, sample_batched in enumerate(data_batched):
        print('Batch no {0}'.format(i_batch))

        X = sample_batched['X']
        Y = sample_batched['Y']

        if cuda:
            X, Y = X.cuda().float(), Y.cuda()

        optimizer.zero_grad()

        if mean_emb == 1:
            Y = torch.mean(Y.float(), dim=1).round()

        output = model1(X)
        # output_temp = output.cpu().numpy()
        # scaler = MinMaxScaler()
        # scaler.fit(output_temp)
        # output_temp = scaler.transform(output_temp)

        # output_temp = torch.from_numpy(  output_temp).cuda()
        # output = F.normalize(output, p=2, dim=1);
        X_prob = model_label(output.detach())
        X_prob = X_prob.view(-1, output_size)

        Y = Y.view(-1, 1).squeeze(1)

        Y = torch.tensor(Y, dtype=torch.long, device=cuda0)

        loss = criterion_CrsEnt(X_prob, Y)

        writer.add_scalar('TrainLoss_Epoch{0}/TrainBatch'.format(ep), loss, i_batch)
        if g_clip > 0:
            torch.nn.utils.clip_grad_norm_(model_label.parameters(), g_clip)

        loss.backward()

        optimizer.step()

        loss_list.append(loss.item())

    total_loss = sum(loss_list)
    writer.add_scalar('Labeled Loss /TrainEpoch', total_loss, ep)
    print('Loss {0} at end of epoch {1}'.format(sum(loss_list), ep))


def train_semisup(pair_batch, label_batch, lam_cp, lam_label, lam_u, ep):
    model.train()
    for param in model.parameters():
        param.requires_grad = True
    loss_list = []

    'Batches for labeled and paired data'
    no_pair = len(pair_batch)
    no_label = len(label_batch)

    pair_batch = list(pair_batch)
    label_batch = list(label_batch)

    for i in list(range(0, max(no_label, no_pair))):

        loss_pair_cp = 0
        loss_pair_label = 0
        optimizer.zero_grad()
        loss_u = 0

        'Training Pair if Pair batch available'
        if i <= no_pair - 1:
            X1 = pair_batch[i]['X1']
            X2 = pair_batch[i]['X2']

            Y_p = pair_batch[i]['Y']
            # X1_label = pair_batch[i]['X1_label']
            # X2_label = pair_batch[i]['X2_label']

            if cuda:
                X1, X2, Y_p = X1.cuda().float(), X2.cuda().float(), Y_p.float().round().cuda()

            # noise_array = torch.empty(X1.shape[0], X1.shape[1], X1.shape[2]).normal_(mean=0, std=1)
            X1_prob = model(X1)
            X2_prob = model(X2)

            kl = KLDiv()

            X1_prob = X1_prob.view(-1, output_size_tcn)
            X2_prob = X2_prob.view(-1, output_size_tcn)

            if mean_emb == 1:
                # Y_p = torch.mean(Y_p.float(), dim=1).round()
                Y_p = torch.mode(Y_p.float(), dim=1)[0].reshape(-1)
                # X1_label_u = torch.mean(X1_label, dim=1).squeeze(1)
                # X2_label_u = torch.mean(X2_label, dim=1).squeeze(1)

            Y_p = Y_p.view(-1, 1).squeeze(1)

            Y_p_pair = torch.tensor(Y_p, dtype=torch.long, device=cuda0)

            kl_div_1 = kl(X1_prob, X2_prob)
            kl_div_2 = kl(X2_prob, X1_prob)

            cost_cp_pair = kl_div_1 + kl_div_2
            loss_pair_cp = hingeloss1(cost_cp_pair, Y_p_pair)

            # X_u_emb = torch.cat((X1_prob, X2_prob),0)
            # loss_u = - torch.sum(torch.sum(torch.mul(X_u_emb,torch.log(X_u_emb)),1))

            if ep == 3999:
                print('pause')
                print('pause at 1000the Epoch')

                X = X1_prob.cpu().detach().numpy()

                Y_l = X1_label.cpu().detach().numpy()

                tsne = TSNE(n_components=2).fit_transform(X)
                X_0 = tsne[Y_l == 0, :]
                X_1 = tsne[Y_l == 1, :]
                X_2 = tsne[Y_l == 2, :]
                X_3 = tsne[Y_l == 3, :]
                X_4 = tsne[Y_l == 4, :]
                X_5 = tsne[Y_l == 5, :]
                plt.scatter(X_0[:, 0], X_0[:, 1], label='0')
                plt.scatter(X_1[:, 0], X_1[:, 1], label='1')
                plt.scatter(X_2[:, 0], X_2[:, 1], label='2')
                plt.scatter(X_3[:, 0], X_3[:, 1], label='3')
                plt.scatter(X_4[:, 0], X_4[:, 1], label='4')
                plt.scatter(X_5[:, 0], X_5[:, 1], label='5')
                plt.title("Represntations on train data")
                plt.legend()
                plt.show()
        'Training Label if Label batch available'
        if i <= no_label - 1:
            X_l = label_batch[i]['X']
            Y_l = label_batch[i]['Y']

            if cuda:
                X_l, Y_l = X_l.cuda().float(), Y_l.float().round().cuda()

            Y_l = torch.mode(Y_l, dim=1)[0].reshape(-1)

            # noise_array = torch.empty(X1.shape[0], X1.shape[1], X1.shape[2]).normal_(mean = 0,std = 1)

            X_prob = model(X_l)
            X_prob = X_prob.view(-1, output_size_tcn)

            'Getting pairs for output probabilities'
            Xl1_prob, Xl2_prob = PairEnum(X_prob)
            Y_l = Y_l.view(-1, 1).squeeze(1)
            'Getting similar/disimilarity indicators for these pairs from class labels'
            Y_p_label = Class2Simi(Y_l)
            # Y_l = torch.tensor(Y_l, dtype=torch.long, device=cuda0)

            kl = KLDiv()

            kl_div_1 = kl(Xl1_prob, Xl2_prob)
            kl_div_2 = kl(Xl2_prob, Xl1_prob)

            cost_pair_label = kl_div_1 + kl_div_2
            loss_pair_label = hingeloss1(cost_pair_label, Y_p_label)

            if ep == 999:
                plot_emb = 1
            else:
                plot_emb = 0

            if plot_emb == 1:
                print('pause')
                print('pause at 1000the Epoch')
                Y_l = Y_l.view(-1, 1)
                Y_l = Y_l.squeeze(1)
                X_labeled = X_prob.cpu().detach().numpy()

                X_unlabeled1 = X1_prob.cpu().detach().numpy()
                X_unlabeled2 = X2_prob.cpu().detach().numpy()

                X = np.vstack((X_labeled, X_unlabeled1, X_unlabeled2))
                Y_l = Y_l.cpu().detach().numpy()

                Y_unl1 = -1 * np.ones(X_unlabeled1.shape[0])
                Y_unl2 = -1 * np.ones(X_unlabeled2.shape[0])
                Y_l_semi = np.append(Y_l, [Y_unl1, Y_unl2])

                tsne = TSNE(n_components=2).fit_transform(X)
                X_0 = tsne[Y_l_semi == 0, :]
                X_1 = tsne[Y_l_semi == 1, :]
                X_2 = tsne[Y_l_semi == 2, :]
                X_3 = tsne[Y_l_semi == 3, :]
                X_4 = tsne[Y_l_semi == 4, :]
                X_6 = tsne[Y_l_semi == -1, :]
                X_5 = tsne[Y_l_semi == 5, :]

                plt.scatter(X_6[:, 0], X_6[:, 1], facecolors='none', edgecolors='black', label='CP pairs')
                plt.scatter(X_0[:, 0], X_0[:, 1], facecolors='none', edgecolors=[(0, 0.4470, 0.7410)], s=100,
                            label='Class 1')
                plt.scatter(X_1[:, 0], X_1[:, 1], facecolors='none', edgecolors=[(0.9290, 0.6940, 0.1250)], s=100,
                            label='Class 2')
                plt.scatter(X_2[:, 0], X_2[:, 1], facecolors='none', edgecolors=[(0.8500, 0.3250, 0.0980)], s=100,
                            label='Class 3')
                plt.scatter(X_3[:, 0], X_3[:, 1], facecolors='none', edgecolors=[(0.4940, 0.1840, 0.5560)], s=100,
                            label='Class 4')
                plt.scatter(X_4[:, 0], X_4[:, 1], edgecolors='g', s=80, label='Class 5')
                plt.scatter(X_5[:, 0], X_5[:, 1], edgecolors='r', s=80, label='Class 6')
                plt.title("Pairwise representations on trained data ", fontweight='bold')
                plt.legend()

                plt.xlabel('t-SNE dim 1')
                plt.ylabel('t-SNE dim 2')

                '''
                plt.savefig('./data_files/gen_simulations/mat_files/mackay_switch/SemiPair_MGlass_20_labels_pairs.pdf',
                            dpi=300 \
                            , format='pdf')
                            '''
                plt.show()

                '''
                X1_l = X1_label_u.cpu().detach().numpy().reshape(-1)
                X2_l = X2_label_u.cpu().detach().numpy().reshape(-1)
                Y_l_all = np.append(Y_l, [X1_l, X2_l])
                X_0 = tsne[ Y_l_all== 0, :]
                X_1 = tsne[Y_l_all == 1, :]
                X_2 = tsne[Y_l_all == 2, :]
                X_3 = tsne[Y_l_all == 3, :]
                X_4 = tsne[Y_l_all == 4, :]

                X_5 = tsne[Y_l_all == 5, :]


                plt.scatter(X_0[:, 0], X_0[:, 1], facecolor = 'none',edgecolors=[(0, 0.4470, 0.7410)], s=80, label='Class 1')
                plt.scatter(X_1[:, 0], X_1[:, 1], facecolor = 'none',edgecolors=[(0.9290, 0.6940, 0.1250)], s=80, label='Class 2')
                plt.scatter(X_2[:, 0], X_2[:, 1],facecolor = 'none', edgecolors=[(0.8500, 0.3250, 0.0980)], s=80, label='Class 3')
                plt.scatter(X_3[:, 0], X_3[:, 1], facecolor = 'none',edgecolors=[(0.4940, 0.1840, 0.5560)], s=80, label='Class 4')
                #plt.scatter(X_4[:, 0], X_4[:, 1], facecolor = 'none',edgecolors='g', s=80, label='Class 5')
                #plt.scatter(X_5[:, 0], X_5[:, 1], facecolor = 'none',edgecolors='r', s=80, label='Class 6')
                plt.title("Pairwise representations - true labels", fontweight='bold')
                plt.xlabel('t-SNE dim 1')
                plt.ylabel('t-SNE dim 2')
                plt.legend(loc='lower left')
                plt.savefig('./data_files/gen_simulations/mat_files/mackay_switch/SemiPair_MGlass_20_labels_true_labels.pdf',
                            dpi=300 \
                            , format='pdf')
                plt.show()
                '''

                '''Y_unl1 = X1_label.cpu().detach().numpy()
                Y_unl2 = X2_label.cpu().detach().numpy()
                Y_l_all = np.append(Y_l, [Y_unl1, Y_unl2])
                X_0 = tsne[Y_l_all == 0, :]
                X_1 = tsne[Y_l_all== 1, :]
                X_2 = tsne[Y_l_all == 2, :]
                X_3 = tsne[Y_l_all == 3, :]
                X_4 = tsne[Y_l_all == 4, :]
                X_5 = tsne[Y_l_all == 5, :]
                plt.scatter(X_0[:, 0], X_0[:, 1], facecolors='none', edgecolors='blue',s=30,label='0')
                plt.scatter(X_1[:, 0], X_1[:, 1], facecolors='none',edgecolors='orange',s=30,label='1')
                plt.scatter(X_2[:, 0], X_2[:, 1], facecolors='none',edgecolors='green',s=30,label='2')
                plt.scatter(X_3[:, 0], X_3[:, 1], facecolors='none',edgecolors='red',s=30,label='3')
                plt.scatter(X_4[:, 0], X_4[:, 1], facecolors='none',edgecolors='purple',s=30,label='4')
                plt.scatter(X_5[:, 0], X_5[:, 1], facecolors='none',edgecolors='brown',s=30,label='5')


                plt.legend()
                plt.title("Represntations for training pairs (labels and cp pairs with all true labels)")
                plt.show()
                '''

        loss = (lam_cp * loss_pair_cp) + (lam_label * loss_pair_label)
        if loss != 0:
            loss.backward()
            optimizer.step()
            if g_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), g_clip)
            loss_list.append(loss.item())
        else:
            loss_list.append(0)

    total_loss = sum(loss_list)
    writer.add_scalar('Pairwise Loss/TrainEpoch', total_loss, ep)
    print('Loss {0} at end of epoch {1}'.format(sum(loss_list), ep))


def train_pairs_dot_prod(ep):
    model.train()
    total_loss = 0
    count = 0
    loss_list = []
    for i_batch, sample_batched in enumerate(pair_train_btch):
        print('Batch no {0}'.format(i_batch))

        X1 = sample_batched['X1']
        X2 = sample_batched['X2']
        Y = sample_batched['Y']

        # X1, X2, Y = increase_all_pairs(X1, X2, Y.squeeze(2))
        if cuda:
            X1, X2, Y = X1.cuda().float(), X2.cuda().float(), Y.cuda()

        optimizer.zero_grad()

        # noise_array = torch.empty(X1.shape[0], X1.shape[1], X1.shape[2]).normal_(mean=0, std=1)
        X1_prob = model(X1)
        X2_prob = model(X2)

        # X1_prob,X2_prob,Y = increase_all_pairs(X1_prob, X2_prob, Y.squeeze(2))
        X1_prob = X1_prob.view(-1, output_size_tcn)
        X2_prob = X2_prob.view(-1, output_size_tcn)

        Y = Y.view(-1, 1)

        Y = Y.squeeze(1)
        loss = criterion_CosSim(X1_prob, X2_prob, Y)

        #   writer.add_scalar('TrainLoss_Epoch{0}/TrainBatch'.format(ep), loss, i_batch)
        if g_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), g_clip)

        loss.backward()

        optimizer.step()

        loss_list.append(loss.item())
    if Y[0] == -1:
        writer.add_histogram('X1 Dissimilar/ epoch', X1_prob[0], ep)
        writer.add_histogram('X2 Dissimilar/ epoch', X2_prob[0], ep)

    total_loss = sum(loss_list)
    writer.add_scalar('Pairwise Loss/TrainEpoch', total_loss, ep)
    print('Loss {0} at end of epoch {1}'.format(sum(loss_list), ep))


def get_no_samples_p_class(train_set):
    list_temp = []
    for i in range(0, len(train_set)):
        list_temp.extend(train_set.dataset[i]['Y'].reshape(-1))
    list_temp = np.array(list_temp)
    per_class_list = []
    for i in range(0, 6):
        per_class_list.append(sum(list_temp == i))
    return per_class_list


def train_semi_sup_ladder(u_batch, l_batch, lam_u, lam_l, ep):
    model_LadderNet.train()
    u_batch = list(u_batch)
    l_batch = list(l_batch)
    no_label = len(l_batch)
    no_unlabel = len(u_batch)
    total_loss = 0
    loss_list = []
    loss_u = 0
    loss_l = 0
    loss_lad_l = 0
    loss_lad_u = 0
    loss_label = 0
    for i in list(range(0, max(no_unlabel, no_label))):
        loss_lad_l = 0
        loss_lad_u = 0
        loss_label = 0

        optimizer_ladder.zero_grad()

        if i <= no_unlabel - 1:
            X = u_batch[i]['X']

            if cuda:
                X = X.cuda().float()

            recon, X1_enc, X1_dec = model_LadderNet(X.transpose(1, 2))
            loss_ladder_temp = 0
            for k in range(0, len(X1_enc)):
                loss_ladder_temp = loss_ladder_temp + criterion_MSE(X1_enc[k], X1_dec[k])
            loss_lad_u = loss_ladder_temp * (1 / len(X1_enc))

        if i <= no_label - 1:
            X = l_batch[i]['X']
            Y = l_batch[i]['Y']
            if cuda:
                X = X.cuda().float()
                Y = Y.cuda().float()

            recon, X1_enc, X1_dec = model_LadderNet(X.transpose(1, 2))
            loss_ladder_temp = 0
            for k in range(0, len(X1_enc)):
                loss_ladder_temp = loss_ladder_temp + criterion_MSE(X1_enc[k], X1_dec[k])
            loss_lad_l = loss_ladder_temp * (1 / len(X1_enc))

            encoding = X1_enc[-1]
            # .view(-1,output_size_tcn) #comment out for AE setting
            encoding = F.normalize(encoding, p=2, dim=1)

            Y = torch.mean(Y.float(), dim=1).round().squeeze(1)

            X_prob = model_label(encoding)

            Y = Y.view(-1, 1).squeeze(1)
            Y = torch.tensor(Y, dtype=torch.long, device=cuda0)

            loss_label = criterion_CrsEnt(X_prob, Y)

        loss = (lam_u * loss_lad_u) + (lam_l * loss_lad_l) + loss_label
        if loss != 0:
            loss.backward()
            optimizer_ladder.step()
            if g_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), g_clip)
            loss_list.append(loss.item())
    total_loss = sum(loss_list)
    writer.add_scalar('AE Semisup Loss/TrainEpoch', total_loss, ep)
    print('Loss {0} at end of epoch {1}'.format(sum(loss_list), ep))


def semi_sup_train_autoenc(u_batch, l_batch, lam_u, lam_l, ep):
    mean_emb = 1
    modelAE.train()
    # temporarily replacing earlier AE with ladder network
    model_label.train()
    total_loss = 0
    count = 0
    loss_list = []

    u_batch = list(u_batch)
    l_batch = list(l_batch)
    'Batches for labeled and paired data'
    no_unlabel = len(u_batch)
    no_label = len(l_batch)
    for i in list(range(0, max(no_unlabel, no_label))):

        loss_ae = 0
        loss_label = 0
        optimizer_ae.zero_grad()

        if i <= no_unlabel - 1:
            X = u_batch[i]['X']
            Y_u = u_batch[i]['Y']
            noise_array = torch.empty(X.shape[0], X.shape[1], X.shape[2]).normal_(mean=0, std=0.01)
            X_noisy = X + noise_array
            # X1, X2, Y = increase_all_pairs(X1, X2, Y.squeeze(2))
            if cuda:
                X = X.cuda().float()
                X_noisy = X_noisy.cuda().float()
            X1_recon, X2_enc = modelAE(X_noisy)
            Y_u = torch.mean(Y_u.float(), dim=1).round().squeeze(1)
            X2_enc = X2_enc.transpose(1, 2)  # .view(-1,output_size_tcn) #comment out for AE setting
            X2_enc = F.normalize(X2_enc, p=2, dim=2)
            if mean_emb == 1:
                X2_enc = torch.mean(X2_enc, dim=1)
            loss_ae = criterion_MSE(X, X1_recon)

        if i <= no_label - 1:
            X = l_batch[i]['X']
            Y = l_batch[i]['Y']
            if cuda:
                X = X.cuda().float()
                Y = Y.cuda().float()
            # Y = Y.view(-1,1)
            recon, encoding = modelAE(X)
            encoding = encoding.transpose(1, 2)  # .view(-1,output_size_tcn) #comment out for AE setting
            encoding = F.normalize(encoding, p=2, dim=2)

            Y = torch.mean(Y.float(), dim=1).round().squeeze(1)
            if mean_emb == 1:
                encoding = torch.mean(encoding, dim=1)

            X_prob = model_label(temp * encoding)

            Y = Y.view(-1, 1).squeeze(1)
            Y = torch.tensor(Y, dtype=torch.long, device=cuda0)

            loss_label = criterion_CrsEnt(X_prob, Y)
            if ep == 999:
                plot_emb = 1
            else:
                plot_emb = 0
            plot_emb = 0
            if plot_emb == 1:
                Y_l = Y.cpu().numpy().reshape(-1)
                Y_u = Y_u.cpu().numpy().reshape(-1)
                encoding = torch.cat((encoding, X2_enc), 0)
                no_un_points = X2_enc.shape[0]
                y_temp = -1 * np.ones(no_un_points)
                Y_u_l = np.concatenate((Y_l, y_temp), axis=0)
                Y_l_l = np.concatenate((Y_l, Y_u), axis=0)

                tsne = TSNE(n_components=2).fit_transform(encoding.detach().cpu().numpy())
                X_u = tsne[Y_u_l == -1, :]
                X_0 = tsne[Y_u_l == 0, :]
                X_1 = tsne[Y_u_l == 1, :]
                X_2 = tsne[Y_u_l == 2, :]
                X_3 = tsne[Y_u_l == 3, :]
                X_4 = tsne[Y_u_l == 4, :]
                X_5 = tsne[Y_u_l == 5, :]
                plt.scatter(X_u[:, 0], X_u[:, 1], facecolor='none', edgecolor='k', label='Unlabelled')
                plt.scatter(X_0[:, 0], X_0[:, 1], facecolor='none', edgecolor=[(0, 0.4470, 0.7410)], label='Class 1',
                            s=100)
                plt.scatter(X_1[:, 0], X_1[:, 1], facecolor='none', edgecolor=[(0.9290, 0.6940, 0.1250)],
                            label='Class 2', s=100)
                plt.scatter(X_2[:, 0], X_2[:, 1], facecolor='none', edgecolor=[(0.8500, 0.3250, 0.0980)] \
                            , label='Class 3', s=100)
                plt.scatter(X_3[:, 0], X_3[:, 1], facecolor='none', edgecolor=[(0.4940, 0.1840, 0.5560)],
                            label='Class 4', s=100)
                # plt.scatter(X_4[:, 0], X_4[:, 1], facecolor='none', edgecolor='g', label='Class 5')
                # plt.scatter(X_5[:, 0], X_5[:, 1], facecolor='none', edgecolor='r', label='Class 6')
                plt.title("Autoencoder representations on trained data ", fontweight='bold')
                plt.legend()

                plt.xlabel('t-SNE dim 1')
                plt.ylabel('t-SNE dim 2')
                plt.savefig('./data_files/gen_simulations/mat_files/mackay_switch/AE_MGlass_20_labels_semi.pdf', dpi=300 \
                            , format='pdf')
                plt.show()

                X_0_u = tsne[Y_l_l == 0, :]
                X_1_u = tsne[Y_l_l == 1, :]
                X_2_u = tsne[Y_l_l == 2, :]
                X_3_u = tsne[Y_l_l == 3, :]
                X_4_u = tsne[Y_l_l == 4, :]
                X_5_u = tsne[Y_l_l == 5, :]
                plt.scatter(X_0_u[:, 0], X_0_u[:, 1], facecolor='none', edgecolor=[(0, 0.4470, 0.7410)],
                            label='Class 1')
                plt.scatter(X_1_u[:, 0], X_1_u[:, 1], facecolor='none', edgecolor=[(0.9290, 0.6940, 0.1250)],
                            label='Class 2')
                plt.scatter(X_2_u[:, 0], X_2_u[:, 1], facecolor='none', edgecolor=[(0.8500, 0.3250, 0.0980)],
                            label='Class 3')
                plt.scatter(X_3_u[:, 0], X_3_u[:, 1], facecolor='none', edgecolor=[(0.4940, 0.1840, 0.5560)],
                            label='Class 4')
                # plt.scatter(X_4_u[:, 0], X_4_u[:, 1], facecolor='none', edgecolor='g',label='Class 5')
                # plt.scatter(X_5_u[:, 0], X_5_u[:, 1], facecolor='none', edgecolor='r',  label='Class 6')
                # plt.scatter(X_0[:, 0], X_0[:, 1], facecolor= [(0, 0.4470, 0.7410)],  s=80)
                # plt.scatter(X_1[:, 0], X_1[:, 1], facecolor= [(0.9290, 0.6940, 0.1250)], s=80)
                # plt.scatter(X_2[:, 0], X_2[:, 1], facecolor=[(0.8500, 0.3250, 0.0980)], s=80)
                # plt.scatter(X_3[:, 0], X_3[:, 1], facecolor=[(0.4940, 0.1840, 0.5560)	], s=80)
                plt.title("Autoencoder representations - true labels", fontweight='bold')
                plt.legend()
                plt.xlabel('t-SNE dim 1')
                plt.ylabel('t-SNE dim 2')
                plt.savefig('./data_files/gen_simulations/mat_files/mackay_switch/AE_MGlass_20_labels_true_lab.pdf',
                            dpi=300 \
                            , format='pdf')
                plt.show()

        loss = (lam_u * loss_ae) + (lam_l * loss_label)
        if loss != 0:
            loss.backward()
            optimizer_ae.step()
            if g_clip > 0:
                torch.nn.utils.clip_grad_norm_(modelAE.parameters(), g_clip)
            loss_list.append(loss.item())
    total_loss = sum(loss_list)
    writer.add_scalar('AE Semisup Loss/TrainEpoch', total_loss, ep)
    print('Loss {0} at end of epoch {1}'.format(sum(loss_list), ep))


def evaluate_semisup_autoenc(eval_batch, f1_score, plot_emb):
    modelAE.train()
    model_label.train()  # Temporarol;y replacing with ladder network
    # model_LadderNet.eval()
    mean_emb = 1
    model_label.eval()
    with torch.no_grad():
        eval_batch = list(eval_batch)
        X = eval_batch[0]['X']
        Y = eval_batch[0]['Y']
        X = X.cuda().float()
        Y_l = np.round(np.mean(Y.cpu().numpy(), axis=1).reshape(-1))
        # Y_l = Y.view(-1,1)
        recon, encoding = modelAE(X)
        encoding = encoding.transpose(1, 2)  # .view(-1,output_size_tcn) comment out for AE setting
        encoding = F.normalize(encoding, p=2, dim=2)
        if mean_emb == 1:
            encoding = torch.mean(encoding, dim=1)
        # Y_l = Y_l.view(-1,1).squeeze(1).cpu().numpy()
        X_prob = model_label(encoding)
        prediction = torch.argmax(X_prob, dim=1).cpu().numpy()
        acc = sum(prediction == Y_l) / len(Y_l)

        # Y_l = Y_l.view(-1, 1)
        # Y_l = Y_l.squeeze(1)

        if plot_emb == 1:
            tsne = TSNE(n_components=2).fit_transform(encoding.cpu().numpy())
            X_0 = tsne[Y_l == 0, :]
            X_1 = tsne[Y_l == 1, :]
            X_2 = tsne[Y_l == 2, :]
            X_3 = tsne[Y_l == 3, :]
            X_4 = tsne[Y_l == 4, :]
            X_5 = tsne[Y_l == 5, :]
            plt.scatter(X_0[:, 0], X_0[:, 1], facecolor='none', edgecolor='b', label='Class 1')
            plt.scatter(X_1[:, 0], X_1[:, 1], facecolor='none', edgecolor='m', label='Class 2')
            plt.scatter(X_2[:, 0], X_2[:, 1], facecolor='none', edgecolor='r', label='Class 3')
            plt.scatter(X_3[:, 0], X_3[:, 1], facecolor='none', edgecolor='g', label='Class 4')
            plt.scatter(X_4[:, 0], X_4[:, 1], facecolor='none', edgecolor='y', label='Class 5')
            plt.scatter(X_5[:, 0], X_5[:, 1], facecolor='none', edgecolor='k', label='Class 6')
            plt.title("Mackay-Glass: Autoencoder representations", fontweight='bold')
            plt.legend()

            plt.xlabel('t-SNE dim 1')
            plt.ylabel('t-SNE dim 2')
            plt.show()

        if f1_score == 1:
            f1_score = get_f1_score(prediction, Y_l, output_size)
            m_f1 = np.mean(f1_score)
            return acc, m_f1
        else:
            return acc


def comp_kern_mmd_train(u_batch, l_batch, lam_u, lam_l, ep):
    mean_emb = 1
    modelAE.train()
    # temporarily replacing earlier AE with ladder network
    model_label.train()
    total_loss = 0
    count = 0
    loss_list = []

    u_batch = list(u_batch)
    l_batch = list(l_batch)
    'Batches for labeled and paired data'
    no_unlabel = len(u_batch)
    no_label = len(l_batch)
    for i in list(range(0, max(no_unlabel, no_label))):

        loss_ae = 0
        loss_label = 0
        optimizer_ae.zero_grad()

        if i <= no_unlabel - 1:
            X = u_batch[i]['X']
            noise_array = torch.empty(X.shape[0], X.shape[1], X.shape[2]).normal_(mean=0, std=1)
            X_noisy = X + noise_array
            # X1, X2, Y = increase_all_pairs(X1, X2, Y.squeeze(2))
            if cuda:
                X = X.cuda().float()
                X_noisy = X_noisy.cuda().float()
            X1_recon, X2_enc = modelAE(X)

            loss_ae = criterion_MSE(X, X1_recon)

        if i <= no_label - 1:
            X = l_batch[i]['X']
            Y = l_batch[i]['Y']
            if cuda:
                X = X.cuda().float()
                Y = Y.cuda().float()
            # Y = Y.view(-1,1)
            recon, encoding = modelAE(X)
            encoding = encoding.transpose(1, 2)  # .view(-1,output_size_tcn) #comment out for AE setting
            encoding = F.normalize(encoding, p=2, dim=2)

            Y = torch.mean(Y.float(), dim=1).round().squeeze(1)
            if mean_emb == 1:
                encoding = torch.mean(encoding, dim=1)

            X_prob = model_label(temp * encoding)

            Y = Y.view(-1, 1).squeeze(1)
            Y = torch.tensor(Y, dtype=torch.long, device=cuda0)

            loss_label = criterion_CrsEnt(X_prob, Y)

        loss = (lam_u * loss_ae) + (lam_l * loss_label)
        if loss != 0:
            loss.backward()
            optimizer_ae.step()
            if g_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), g_clip)
            loss_list.append(loss.item())
    total_loss = sum(loss_list)
    writer.add_scalar('AE Semisup Loss/TrainEpoch', total_loss, ep)
    print('Loss {0} at end of epoch {1}'.format(sum(loss_list), ep))


def comp_kern_mmd_eval(eval_batch, f1_score):
    modelAE.eval()  # Temporarol;y replacing with ladder network
    # model_LadderNet.eval()
    mean_emb = 1
    model_label.eval()
    with torch.no_grad():
        eval_batch = list(eval_batch)
        X = eval_batch[0]['X']
        Y = eval_batch[0]['Y']
        X = X.cuda().float()
        Y_l = np.round(np.mean(Y.cpu().numpy(), axis=1).reshape(-1))
        # Y_l = Y.view(-1,1)
        recon, encoding = modelAE(X)
        encoding = encoding.transpose(1, 2)  # .view(-1,output_size_tcn) comment out for AE setting
        encoding = F.normalize(encoding, p=2, dim=2)
        if mean_emb == 1:
            encoding = torch.mean(encoding, dim=1)
        # Y_l = Y_l.view(-1,1).squeeze(1).cpu().numpy()
        X_prob = model_label(encoding)
        prediction = torch.argmax(X_prob, dim=1).cpu().numpy()
        acc = sum(prediction == Y_l) / len(Y_l)

        # Y_l = Y_l.view(-1, 1)
        # Y_l = Y_l.squeeze(1)

        plot_emb = 0
        if plot_emb == 1:
            tsne = TSNE(n_components=2).fit_transform(encoding.cpu().numpy())
            X_0 = tsne[Y_l == 0, :]
            X_1 = tsne[Y_l == 1, :]
            X_2 = tsne[Y_l == 2, :]
            X_3 = tsne[Y_l == 3, :]

            plt.scatter(X_0[:, 0], X_0[:, 1], facecolor='none', edgecolor='b', label='0')
            plt.scatter(X_1[:, 0], X_1[:, 1], facecolor='none', edgecolor='g', label='0')
            plt.scatter(X_2[:, 0], X_2[:, 1], facecolor='none', edgecolor='r', label='0')
            plt.scatter(X_3[:, 0], X_3[:, 1], facecolor='none', edgecolor='m', label='0')

            plt.title("Representations for test data")
            plt.legend()
            plt.title('T-SNE on Encoded Space of dim10 of AutoEnc')
            plt.show()
        if f1_score == 1:
            f1_score = get_f1_score(prediction, Y_l, output_size)
            m_f1 = np.mean(f1_score)
            return acc, m_f1
        else:
            return acc


def eval_autoenc(data_eval, ep):
    modelAE.eval()
    total_loss = 0
    count = 0
    loss_list = []
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(data_eval):
            print('Batch no {0}'.format(i_batch))

            X = sample_batched['X']
            Y = sample_batched['Y']
            noise_array = torch.empty(X.shape[0], X.shape[1], X.shape[2]).normal_(mean=0, std=0.1)
            X_noisy = X + noise_array

            if cuda:
                X = X.cuda().float()
                X_noisy = X_noisy.cuda().float()

            # noise_array = torch.empty(X1.shape[0], X1.shape[1], X1.shape[2]).normal_(mean=0, std=1)
            X1_recon, X2_enc = modelAE(X_noisy)

            loss = criterion_MSE(X, X1_recon)

            if ep == 1400:
                print('pause')
                print('pause at 1000the Epoch')
                Y = Y.view(-1, 1)
                Y = Y.squeeze(1)
                X2_enc = X2_enc.transpose(1, 2).view(-1, output_size_tcn)
                X = X2_enc.cpu().detach().numpy()
                Y = Y.cpu().detach().numpy()
                tsne = TSNE(n_components=2).fit_transform(X)
                X_0 = tsne[Y == 0, :]
                X_1 = tsne[Y == 1, :]
                X_2 = tsne[Y == 2, :]
                X_3 = tsne[Y == 3, :]
                X_4 = tsne[Y == 4, :]
                X_5 = tsne[Y == 5, :]
                plt.scatter(X_0[:, 0], X_0[:, 1], label='0')
                plt.scatter(X_1[:, 0], X_1[:, 1], label='1')
                plt.scatter(X_2[:, 0], X_2[:, 1], label='2')
                plt.scatter(X_3[:, 0], X_3[:, 1], label='3')
                plt.scatter(X_4[:, 0], X_4[:, 1], label='4')
                plt.scatter(X_5[:, 0], X_5[:, 1], label='5')
                plt.legend()
                plt.title('T-SNE on Encoded Space of dim10 of AutoEnc')
                # sio.savemat('./low_dim_autp_encmore_diff_tsne.mat',
                #            {'X0': X_0, 'X1': X_1, 'X2': X_2, 'X3': X_3, 'X4': X_4, 'X5': X_5})
                # plt.savefig('./tsne_autoenc.png')

            loss_list.append(loss.item())

    total_loss = sum(loss_list)
    writer.add_scalar('Vlaidation AE loss', total_loss, ep)
    print('Loss {0} at end of epoch {1}'.format(sum(loss_list), ep))
    return total_loss


def eval_ladder(data_eval, lam_u):
    model_LadderNet.eval()
    total_loss = 0
    count = 0
    loss_list = []
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(data_eval):
            print('Batch no {0}'.format(i_batch))

            X = sample_batched['X']
            Y = sample_batched['Y']

            if cuda:
                X = X.cuda().float()

            # noise_array = torch.empty(X1.shape[0], X1.shape[1], X1.shape[2]).normal_(mean=0, std=1)
            X1_recon, X1_enc, X1_dec = model_LadderNet(X.transpose(1, 2))

            loss_ladder_temp = 0
            for k in range(0, len(X1_enc)):
                loss_ladder_temp = loss_ladder_temp + criterion_MSE(X1_enc[k], X1_dec[k])
            loss_lad_u = loss_ladder_temp * (1 / len(X1_enc))

            encoding = X1_enc[-1]
            # .view(-1,output_size_tcn) #comment out for AE setting
            encoding = F.normalize(encoding, p=2, dim=1)

            Y = torch.mean(Y.float(), dim=1).round().squeeze(1)

            X_prob = model_label(encoding)

            Y = Y.view(-1, 1).squeeze(1)
            Y = torch.tensor(Y, dtype=torch.long, device=cuda0)

            loss_label = criterion_CrsEnt(X_prob, Y)
            loss_list.append(loss_label)

            loss_list.append(lam_u * loss_lad_u + loss_label)

        total_loss = sum(loss_list)
        return total_loss


def eval_semisup_ladder(eval_batch, plot_emb, f1_score):
    model_LadderNet.eval()
    total_loss = 0
    count = 0
    loss_list = []
    mean_emb = 1
    model_label.eval()
    with torch.no_grad():
        eval_batch = list(eval_batch)
        X = eval_batch[0]['X']
        Y = eval_batch[0]['Y']
        X = X.cuda().float()
        Y_l = np.round(np.mean(Y.cpu().numpy(), axis=1).reshape(-1))

        if cuda:
            X = X.cuda().float()

        X1_recon, X1_enc, X1_dec = model_LadderNet(X.transpose(1, 2))

        encoding = X1_enc[-1]

        encoding = F.normalize(encoding, p=2, dim=1)

        mean_emb = 1

        X_prob = model_label(temp * encoding)

        Y = Y.view(-1, 1).squeeze(1)

        prediction = torch.argmax(X_prob, dim=1).cpu().numpy()
        acc = sum(prediction == Y_l) / len(Y_l)

        # Y_l = Y_l.view(-1, 1)
        # Y_l = Y_l.squeeze(1)

        if plot_emb == 1:
            tsne = TSNE(n_components=2).fit_transform(encoding.cpu().numpy())
            X_0 = tsne[Y_l == 0, :]
            X_1 = tsne[Y_l == 1, :]
            X_2 = tsne[Y_l == 2, :]
            X_3 = tsne[Y_l == 3, :]
            X_4 = tsne[Y_l == 4, :]
            plt.scatter(X_0[:, 0], X_0[:, 1], facecolor='none', edgecolor='b', label='0')
            plt.scatter(X_1[:, 0], X_1[:, 1], facecolor='none', edgecolor='m', label='0')
            plt.scatter(X_2[:, 0], X_2[:, 1], facecolor='none', edgecolor='r', label='0')
            plt.scatter(X_3[:, 0], X_3[:, 1], facecolor='none', edgecolor='g', label='0')
            plt.scatter(X_4[:, 0], X_4[:, 1], facecolor='none', edgecolor='y', label='0')
            plt.title("Representations for test data")
            plt.legend()
            plt.title('T-SNE on Encoded Space of dim10 of AutoEnc')
            plt.show()
        if f1_score == 1:
            f1_score = get_f1_score(prediction, Y_l, output_size)
            m_f1 = np.mean(f1_score)
            return acc, m_f1
        else:
            return acc


def eval_pairs_dot_product(data_eval):
    model.eval()
    loss_list = []
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(data_eval):

            X1 = sample_batched['X1']
            X2 = sample_batched['X2']
            Y = sample_batched['Y']
            # X1, X2, Y = increase_all_pairs(X1, X2, Y.squeeze(2))
            if cuda:
                X1, X2, Y = X1.cuda().float(), X2.cuda().float(), Y.cuda()

            X1_prob = model(X1)
            X2_prob = model(X2)

            # X1_prob_avg = torch.mean(X1_prob, dim=1)
            # X2_prob_avg = torch.mean(X2_prob, dim=1)
            # Y = torch.mean(Y, dim =1 )
            # X1_prob, X2_prob, Y = increase_all_pairs(X1_prob, X2_prob, Y.squeeze(2))
            X1_prob = X1_prob.view(-1, output_size_tcn)
            X2_prob = X2_prob.view(-1, output_size_tcn)

            Y = Y.view(-1, 1)
            Y = torch.tensor(Y, dtype=torch.long, device=cuda0)
            Y = Y.squeeze(1)

            loss = criterion_CosSim(X1_prob, X2_prob, Y)
            loss_list.append(loss)

    return sum(loss_list)


def evaluate_labels(data_eval):
    model.eval()
    model_label.eval()
    pred_list = []
    Y_list = []
    loss_list = []
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(data_eval):

            X = sample_batched['X']
            Y = sample_batched['Y']

            if cuda:
                X, Y = X.cuda().float(), Y.cuda()

            output = model1(X)
            X_prob = model_label(output)

            if mean_emb == 1:
                Y = torch.mean(Y.float(), dim=1).round()

            X_prob = X_prob.view(-1, output_size)
            Y = Y.view(-1, 1) - 1

            'Taking average across all points'
            pred = torch.argmax(X_prob, dim=1)

            Y = torch.tensor(Y, dtype=torch.long, device=cuda0)
            Y = Y.squeeze(1)
            acc = torch.sum(Y.squeeze(0) == pred).data / float(len(Y))
            pred = pred.cpu().numpy()
            Y = Y.cpu().numpy()

            Y_list.extend(Y.tolist())
            pred_list.extend(pred.tolist())
            loss_list.append(acc)
    # m = MultiLabelBinarizer().fit(np.array(Y_list))
    # k = MultiLabelBinarizer().fit(np.array(pred_list))
    # f_score = f1_score(m.transform(np.array(Y_list)), m.transform(np.array(pred_list)), average='weighted')
    # f_score = get_f1_score(pred_list,Y_list,6)
    return sum(loss_list) / len(loss_list)


def evaluate_pairs(data_eval):
    model.eval()
    loss_list = []
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(data_eval):

            X1 = sample_batched['X1']
            X2 = sample_batched['X2']
            # X1_label = sample_batched['X1_label']
            # X2_label = sample_batched['X2_label']
            Y = sample_batched['Y']

            if cuda:
                X1, X2, Y = X1.cuda().float(), X2.cuda().float(), Y.cuda().round()

            if mean_emb == 1:
                Y = torch.mean(Y, dim=1).round()
                # X1_label = torch.mean(X1_label, dim=1).squeeze(1)
                # X2_label = torch.mean(X2_label, dim=1).squeeze(1)

            X1_prob = model(X1)
            X2_prob = model(X2)

            # X1_prob, X2_prob, Y = increase_all_pairs(X1_prob, X2_prob, Y.squeeze(2))
            X1_prob = X1_prob.view(-1, output_size_tcn)
            X2_prob = X2_prob.view(-1, output_size_tcn)
            Y = Y.view(-1, 1).squeeze(1)
            Y = torch.tensor(Y, dtype=torch.long, device=cuda0)

            kl = KLDiv()

            kl_div_1 = kl(X1_prob, X2_prob)
            kl_div_2 = kl(X2_prob, X1_prob)

            Y = torch.tensor(Y, dtype=torch.long, device=cuda0)

            cost = kl_div_1 + kl_div_2

            loss = (hingeloss1(cost, Y))

            loss_list.append(loss)
            eval_pairs = 0
            if eval_pairs == 1:
                tsne = TSNE(n_components=2).fit_transform(X1_prob.cpu().detach().numpy())
                X1_label = sample_batched['X1_label']
                X1_label = torch.mean(X1_label, dim=1).squeeze(1)
                label_list = X1_label.unique().cpu().tolist()
                for i in label_list:
                    X_0 = tsne[X1_label == i, :]
                    plt.scatter(X_0[:, 0], X_0[:, 1], label=str(i))

                plt.legend()
                plt.title('T-SNE on Encoded Space From Pairs from CP')
                plt.show()

    return sum(loss_list)


def train_feedforward_semi_vat(l_batch, u_batch, lam_l, lam_u):
    mean_emb = 1

    # temporarily replacing earlier AE with ladder network
    model_label.train()
    model.train()
    for param in model.parameters():
        param.requires_grad = False

    count = 0
    loss_list = []

    u_batch = list(u_batch)
    l_batch = list(l_batch)
    'Batches for labeled and paired data'
    no_unlabel = len(u_batch)
    # no_unlabel = 0
    no_label = len(l_batch)

    for i in list(range(0, max(no_unlabel, no_label))):

        loss_label = 0
        loss_pi_unlb = 0
        loss_pi_lab = 0
        optimizer_classifier.zero_grad()

        if i <= no_unlabel - 1:
            X_u = u_batch[i]['X1']
            # noise_array = torch.empty(X_u.shape[0], X.shape[1], X.shape[2]).normal_(mean=0, std=1)

            # X_noisy = X + noise_array
            # X1, X2, Y = increase_all_pairs(X1, X2, Y.squeeze(2))
            if cuda:
                X_u = X_u.cuda().float()
                rand_u = torch.empty(X_u.shape[0], X_u.shape[1], X_u.shape[2]).normal_(mean=0.0, std=0.01).cuda()
            else:
                rand_u = torch.empty(X_u.shape[0], X_u.shape[1], X_u.shape[2]).normal_(mean=0.0, std=0.01)
            rand_u.requires_grad = True
            emb1 = model(X_u)
            emb_perturb = model(X_u + rand_u)
            X_u_prob = model_label(emb1)
            X_u_prob_pert = model_label(emb_perturb)

            for param in model_label.parameters():
                param.requires_grad = False
            loss_emb = kl_div1(torch.log(X_u_prob.detach()), X_u_prob_pert)
            loss_emb.backward()

            r_ad_u = Variable(F.normalize((rand_u.grad.data.clone()), 2))

            optimizer_classifier.zero_grad()

            for param in model_label.parameters():
                param.requires_grad = True
            X_u_prob = model_label(model(X_u))
            X_u_prob_pert = model_label(model(X_u + r_ad_u))
            loss_pi_unlb = kl_div1(torch.log(X_u_prob.detach()), X_u_prob_pert)

        if i <= no_label - 1:
            X_l = l_batch[i]['X']
            Y = l_batch[i]['Y']

            if cuda:
                X_l = X_l.cuda().float()
                Y = Y.cuda().float()
                rand_l = torch.empty(X_l.shape[0], X_l.shape[1], X_l.shape[2]).normal_(mean=0.0, std=0.01).cuda()
            else:
                rand_l = torch.empty(X_l.shape[0], X_l.shape[1], X_l.shape[2]).torch.normal(mean=0.0, std=0.01)
            # Y = Y.view(-1,1)
            rand_l.requires_grad = True

            for param in model_label.parameters():
                param.requires_grad = False
            X_l_prob = model_label(model(X_l))
            X_l_prob_perturbed = model_label(model(X_l + rand_l))

            loss_ad_lab = kl_div2(torch.log(X_l_prob.detach()), X_l_prob_perturbed)
            loss_ad_lab.backward()
            r_ad_l = Variable(F.normalize((rand_l.grad.data.clone()), 2))
            optimizer_classifier.zero_grad()

            for param in model_label.parameters():
                param.requires_grad = True

            X_l_prob = model_label(model(X_l))
            X_l_prob_perturbed = model_label(model(X_l + r_ad_l))

            loss_pi_lab = kl_div2(torch.log(X_l_prob.detach()), X_l_prob_perturbed)

            Y = torch.mean(Y.float(), dim=1).round().squeeze(1)

            # if mean_emb == 1:

            X_l_prob.requies_grad = True
            Y = Y.view(-1, 1).squeeze(1)
            Y = torch.tensor(Y, dtype=torch.long, device=cuda0)
            loss_label = criterion_CrsEnt(X_l_prob, Y)

        loss = (lam_u * loss_pi_unlb) + (lam_l * loss_pi_lab) + loss_label
        # loss = loss_label
        if loss != 0:
            loss.backward()
            optimizer_classifier.step()
            if g_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), g_clip)
            loss_list.append(loss.item())

    total_loss = sum(loss_list)
    writer.add_scalar('SemiSup Pi Feeedforwad loss', total_loss, ep)
    print('Loss {0} at end of epoch {1}'.format(sum(loss_list), ep))


def train_semi_entrop_reg_feedforward(l_batch, u_batch, lam_l, lam_u):
    'This script implements a semisup feed forward classifier using entropy regularization'
    mean_emb = 1
    'Freeze layers for pairwise representations'
    for param in model.parameters():
        param.requires_grad = False
    # temporarily replacing earlier AE with ladder network
    model_label.train()
    total_loss = 0
    count = 0
    loss_list = []

    u_batch = list(u_batch)
    l_batch = list(l_batch)
    'Batches for labeled and paired data'
    no_unlabel = len(u_batch)
    no_label = len(l_batch)
    # for i in list(range(0, max(no_unlabel, no_label))):

    loss_u = 0
    loss_label = 0
    optimizer_classifier.zero_grad()
    loss = 0
    'unlabelled batches'
    # if i <= no_unlabel - 1:
    X1 = u_batch[0]['X1']
    X2 = u_batch[0]['X2']

    if cuda:
        X1 = X1.cuda().float()
        X2 = X2.cuda().float()
    X1_emb = model(X1)
    X2_emb = model(X2)

    X_u_e = torch.cat((X1_emb, X2_emb), 0)

    X_u = model_label(X_u_e)

    loss_u = - (1 / X_u.shape[0]) * torch.sum(torch.sum(torch.mul(X_u, torch.log(X_u)), 1))

    # if i <= no_label - 1:
    X = l_batch[0]['X']
    Y = l_batch[0]['Y']
    if cuda:
        X = X.cuda().float()
        Y = Y.cuda().float()

    X_enc = model(X)
    if mean_emb == 1:
        Y = torch.mean(Y.float(), dim=1).round().squeeze(1)

    X_prob = model_label(X_enc)
    Y_l = Y.view(-1, 1).squeeze(1)
    Y_l = torch.tensor(Y, dtype=torch.long, device=cuda0)

    loss_label = criterion_CrsEnt(X_prob, Y_l)

    loss = (lam_u * loss_u) + (lam_l * loss_label)
    if loss != 0:
        loss.backward()
        optimizer_classifier.step()
        if g_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), g_clip)
        loss_list.append(loss.item())
    total_loss = sum(loss_list)
    writer.add_scalar('AE Semisup Loss/TrainEpoch', total_loss, ep)
    print('Loss {0} at end of epoch {1}'.format(sum(loss_list), ep))

    plot_emb = 0
    if plot_emb == 1:
        print('pause')
        print('pause at 1000the Epoch')
        Y_l = Y_l.view(-1, 1)
        Y_l = Y_l.squeeze(1)
        X_labeled = X_prob.cpu().detach().numpy()
        X_unlabeled1 = X_u.cpu().detach().numpy()

        X = np.vstack((X_labeled, X_unlabeled1))
        Y_l = Y_l.cpu().detach().numpy()
        Y_unl1 = -1 * np.ones(X_unlabeled1.shape[0])

        Y_l_semi = np.append(Y_l, Y_unl1)

        tsne = TSNE(n_components=2).fit_transform(X)
        X_0 = tsne[Y_l_semi == 0, :]
        X_1 = tsne[Y_l_semi == 1, :]
        X_2 = tsne[Y_l_semi == 2, :]
        X_3 = tsne[Y_l_semi == 3, :]
        X_4 = tsne[Y_l_semi == 4, :]
        X_6 = tsne[Y_l_semi == -1, :]
        X_5 = tsne[Y_l_semi == 5, :]

        plt.scatter(X_6[:, 0], X_6[:, 1], facecolors='none', edgecolors='black', label='CP pairs')
        plt.scatter(X_0[:, 0], X_0[:, 1], facecolors=[(0, 0.4470, 0.7410)], s=80, label='Class 1')
        plt.scatter(X_1[:, 0], X_1[:, 1], facecolors=[(0.9290, 0.6940, 0.1250)], s=80, label='Class 2')
        plt.scatter(X_2[:, 0], X_2[:, 1], facecolors=[(0.8500, 0.3250, 0.0980)], s=80, label='Class 3')
        plt.scatter(X_3[:, 0], X_3[:, 1], facecolors=[(0.4940, 0.1840, 0.5560)], s=80, label='Class 4')
        plt.scatter(X_4[:, 0], X_4[:, 1], facecolors='g', s=80, label='Class 5')
        plt.scatter(X_5[:, 0], X_5[:, 1], facecolors='r', s=80, label='Class 6')
        plt.title("Pairwise representations on trained data ", fontweight='bold')
        plt.legend()

        plt.xlabel('t-SNE dim 1')
        plt.ylabel('t-SNE dim 2')

        plt.legend()
        plt.show()


def train_feedforward(train_data_batch, semi_sup):
    model_label.train()

    # model.train()
    # model1.load_state_dict(torch.load('./models/parwise_good_mean_shift'))

    'Freeze layers for pairwise representations'
    for param in model.parameters():
        param.requires_grad = False

    for i_batch, sample_batched in enumerate(train_data_batch):
        print('Batch no {0}'.format(i_batch))

        optimizer_classifier.zero_grad()

        X = sample_batched['X']
        Y = sample_batched['Y']

        if cuda:
            X, Y = X.cuda().float(), Y.cuda()

        if semi_sup == 0:
            Y = torch.mean(Y.float(), dim=1).round()

        'Getting model representation for labels'
        output = model(X)

        # output = F.softmax(output,dim=1)
        # output_temp = output.cpu().numpy()
        # scaler = MinMaxScaler()
        # scaler.fit(output_temp)
        # output_temp = scaler.transform(output_temp)

        # output_temp = torch.from_numpy(  output_temp).cuda()
        # output = F.normalize(output, p=2, dim=1);
        'Train c'
        X_prob = model_label(output.float())

        X_prob = X_prob.view(-1, output_size)

        Y = Y.view(-1, 1).squeeze(1)

        Y = torch.tensor(Y, dtype=torch.long, device=cuda0)

        loss = criterion_CrsEnt(X_prob, Y)

        if g_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), g_clip)

        loss.backward()

        optimizer_classifier.step()
    print('Loss {0} at end of epoch {1}'.format(loss, ep))
    return loss


def evaluate_classification(eval_data, f1_score, plot):
    model1.eval()
    model_label.eval()
    model.eval()
    with torch.no_grad():
        temp = list(eval_data)
        X = temp[0]['X']
        Y = temp[0]['Y']
        X = X.cuda().float()
        Y_l = mode(Y.cpu().numpy(), axis=1).mode.reshape(-1).astype(int)
        # if mean_emb == 1:
        #    Y_l = torch.mean(Y.float(), dim=1).round()
        X_prob = model(X)
        representation = X_prob.view(-1, output_size_tcn)
        prediction = torch.argmax(model_label(representation), dim=1).cpu().numpy()
        acc = sum(prediction == Y_l) / len(Y_l)
        if plot == 1:
            tsne = TSNE(n_components=2).fit_transform(representation.cpu().numpy())
            X_0 = tsne[Y_l == 0, :]
            X_1 = tsne[Y_l == 1, :]
            X_2 = tsne[Y_l == 2, :]
            X_3 = tsne[Y_l == 3, :]
            X_4 = tsne[Y_l == 4, :]
            X_5 = tsne[Y_l == 5, :]
            plt.scatter(X_0[:, 0], X_0[:, 1], label='0')
            plt.scatter(X_1[:, 0], X_1[:, 1], label='1')
            plt.scatter(X_2[:, 0], X_2[:, 1], label='2')
            plt.scatter(X_3[:, 0], X_3[:, 1], facecolor='none', edgecolors='k', label='3')
            plt.scatter(X_4[:, 0], X_4[:, 1], label='4')
            plt.scatter(X_5[:, 0], X_5[:, 1], label='5')
            plt.title("Representations learned with 30 labels")
            plt.legend()
            # plt.title('T-SNE on Encoded Space of dim10 of AutoEnc')
            plt.show()

        # f_score = get_f1_score(prediction, Y_l, 6)
        # Y_l = Y_l.view(-1, 1)
        # Y_l = Y_l.squeeze(1)

        if f1_score == 1:
            f1_score = get_f1_score(prediction, Y_l, output_size)
            m_f1 = np.mean(f1_score)
            return acc, m_f1
        else:
            return acc


def evaluate_classification_uni_batch(eval_data, f1_score, plot):
    model1.eval()
    model_label.eval()
    rep_array = np.array([])
    prediction_array = np.array([])
    label_array = np.array([])
    # model.train()
    with torch.no_grad():
        temp = list(eval_data)
        for i in range(0, len(temp)):
            X = temp[i]['X']
            Y = temp[i]['Y']
            X = X.cuda().float()
            Y_l = mode(Y.cpu().numpy(), axis=1).mode.reshape(-1).astype(int)
            # if mean_emb == 1:
            #    Y_l = torch.mean(Y.float(), dim=1).round()
            X_prob = model(X)
            representation = X_prob.view(-1, output_size_tcn)
            prediction = torch.argmax(model_label(representation), dim=1).cpu().numpy()
            prediction_array = np.concatenate((prediction_array, prediction), axis=0) if prediction_array.size \
                else prediction
            label_array = np.concatenate((label_array, Y_l), axis=0) if label_array.size \
                else Y_l

        acc = sum(prediction_array == label_array) / len(label_array)
        if plot == 1:
            tsne = TSNE(n_components=2).fit_transform(representation.cpu().numpy())
            X_0 = tsne[Y_l == 0, :]
            X_1 = tsne[Y_l == 1, :]
            X_2 = tsne[Y_l == 2, :]
            X_3 = tsne[Y_l == 3, :]
            X_4 = tsne[Y_l == 4, :]
            X_5 = tsne[Y_l == 5, :]
            plt.scatter(X_0[:, 0], X_0[:, 1], label='0')
            plt.scatter(X_1[:, 0], X_1[:, 1], label='1')
            plt.scatter(X_2[:, 0], X_2[:, 1], label='2')
            plt.scatter(X_3[:, 0], X_3[:, 1], label='3')
            plt.scatter(X_4[:, 0], X_4[:, 1], label='4')
            plt.scatter(X_5[:, 0], X_5[:, 1], label='5')
            plt.title("Representations learned with 30 labels")
            plt.legend()
            # plt.title('T-SNE on Encoded Space of dim10 of AutoEnc')
            plt.show()

        # f_score = get_f1_score(prediction, Y_l, 6)
        # Y_l = Y_l.view(-1, 1)
        # Y_l = Y_l.squeeze(1)

        if f1_score == 1:
            f1_score = get_f1_score(prediction_array, label_array, output_size)
            m_f1 = np.mean(f1_score)
            return acc, m_f1
        else:
            return acc


def label_propogate(X_l, X_u, Y_l):
    model.eval()
    with torch.no_grad():
        X_l = model(X_l).cpu().detach().numpy()
        X_u = model(X_u).cpu().detach().numpy()
        X = np.vstack((X_l, X_u))
        Y_l = Y_l.cpu().detach().numpy()
        Y_unl1 = -1 * np.ones(X_u.shape[0])
        Y_l_semi = np.append(Y_l, Y_unl1)
        label_prop_model = LabelPropagation(gamma=40)
        label_prop_model.fit(X, Y_l_semi)
        Y_prop = label_prop_model.transduction_

        tsne = TSNE(n_components=2).fit_transform(X)
        X_0 = tsne[Y_l_semi == 0, :]
        X_1 = tsne[Y_l_semi == 1, :]
        X_2 = tsne[Y_l_semi == 2, :]
        X_3 = tsne[Y_l_semi == 3, :]
        X_4 = tsne[Y_l_semi == 4, :]
        X_5 = tsne[Y_l_semi == 5, :]
        X_6 = tsne[Y_l_semi == -1, :]
        plt.scatter(X_6[:, 0], X_6[:, 1], facecolors='none', edgecolors='black', label='unlabelled')
        plt.scatter(X_0[:, 0], X_0[:, 1], s=50, label='0')
        plt.scatter(X_1[:, 0], X_1[:, 1], s=50, label='1')
        plt.scatter(X_2[:, 0], X_2[:, 1], s=50, label='2')
        plt.scatter(X_3[:, 0], X_3[:, 1], s=50, label='3')
        plt.scatter(X_4[:, 0], X_4[:, 1], s=50, label='4')
        plt.scatter(X_5[:, 0], X_5[:, 1], s=50, label='5')
        plt.title("Represntations for training pairs (labels and from cps)")
        plt.legend()
        plt.show()

        X_0 = tsne[Y_prop == 0, :]
        X_1 = tsne[Y_prop == 1, :]
        X_2 = tsne[Y_prop == 2, :]
        X_3 = tsne[Y_prop == 3, :]
        X_4 = tsne[Y_prop == 4, :]
        X_5 = tsne[Y_prop == 5, :]

        plt.scatter(X_0[:, 0], X_0[:, 1], s=50, label='0')
        plt.scatter(X_1[:, 0], X_1[:, 1], s=50, label='1')
        plt.scatter(X_2[:, 0], X_2[:, 1], s=50, label='2')
        plt.scatter(X_3[:, 0], X_3[:, 1], s=50, label='3')
        plt.scatter(X_4[:, 0], X_4[:, 1], s=50, label='4')
        plt.scatter(X_5[:, 0], X_5[:, 1], s=50, label='5')
        plt.title("Represntations for training pairs (labels and from cps)")
        plt.legend()
        plt.show()
        return Y_prop


0

if __name__ == "__main__":
    balanced_train_set = get_balanced_samples(train_set_lab, train_samples, output_size)
    val_balanced_test = get_balanced_samples(train_set_lab, 1000, output_size)

    torch.save(modelAE.state_dict(), '/home/nahad/pycharm_mac/semi_with_cp/semi_ts_clustering/models/temp_model_AE')
    torch.save(model_label.state_dict(), '/home/nahad/pycharm_mac/semi_with_cp/semi_ts_clustering/models/temp_label')
    #torch.save(model.state_dict(), '/home/nahad/pycharm_mac/semi_with_cp/semi_ts_clustering/models/temp_model')


    ae_result_list = []
    semi_cp_result_list = []
    sup_result_list = []
    for k in range(1,100):
        train_set = balanced_train_set[k]
        balanced_train_btch = DataLoader(train_set, batch_size=batch_size_label, shuffle=True)
        balanced_val_btch = DataLoader(val_balanced_test, batch_size=1000, shuffle=True)

        #
        # model1.load_state_dict(torch.load('./models/with_cp_and_labels_har'))
        # model1.load_state_dict(torch.load('./models/parwise_good_mean_shift'))
        # model.load_state_dict(torch.load('./models/ar_shift_50_800_cp_pairs'))
        # model.load_state_dict(torch.load('./models/AR_50_label_cp_pairs'))
        # model.load_state_dict(torch.load('./models/HAR/ActiTraker/semisup_cp_lab_50'))
        # model.load_state_dict(torch.load('/home/nahad/pycharm_mac/semi_with_cp/semi_ts_clustering/models/HAR/ActiTraker/semisup_cp_lab_50'))
        # model.load_state_dict(torch.load('/home/nahad/pycharm_mac/semi_with_cp/semi_ts_clustering/models/simulations/AR_mean_var/AE/AE_labels_5'))

        #model.load_state_dict( torch.load('/home/nahad/pycharm_mac/semi_with_cp/semi_ts_clustering/models/temp_model'))
        model_label.load_state_dict(torch.load('/home/nahad/pycharm_mac/semi_with_cp/semi_ts_clustering/models/temp_label'))
        modelAE.load_state_dict(torch.load('/home/nahad/pycharm_mac/semi_with_cp/semi_ts_clustering/models/temp_model_AE'))

        'training ae'

        for ep in range(0, 600):
            print('Labelled Epoch number: {0}'.format(ep))
            'Lambda for cp pairs'

            lam_ul = 0.1
            lam_l = 1
            semi_sup_train_autoenc(ul_data_batch, balanced_train_btch, lam_ul, lam_l, ep)
            print('Loss AE for loop number{0}'.format(k))
            acc = evaluate_semisup_autoenc(label_val_btch, 0, 0)
            writer.add_scalar('Loss De noising autoencoder', acc, ep)

        acc_temp = evaluate_semisup_autoenc(label_data_test, 1, 0)
        ae_result_list.append(acc_temp)

        'training sup'
        model_label.load_state_dict(torch.load('/home/nahad/pycharm_mac/semi_with_cp/semi_ts_clustering/models/temp_label'))
        modelAE.load_state_dict(torch.load('/home/nahad/pycharm_mac/semi_with_cp/semi_ts_clustering/models/temp_model_AE'))

        for ep in range(0, 600):
            print('Labelled Epoch number: {0}'.format(ep))
            'Lambda for cp pairs'
            lam_ul = 0
            lam_l = 1
            semi_sup_train_autoenc(ul_data_batch, balanced_train_btch, lam_ul, lam_l, ep)
            acc = evaluate_semisup_autoenc(label_val_btch, 0, 0)
            print('Loss supervised for loop number{0}'.format(k))
            writer.add_scalar('Loss De noising autoencoder', acc, ep)

        sup_temp = evaluate_semisup_autoenc(label_data_test, 1, 0)
        sup_result_list.append(sup_temp)


        'training semi sup cp'

        modelAE.load_state_dict(torch.load('/home/nahad/pycharm_mac/semi_with_cp/semi_ts_clustering/models/temp_model_AE'))
        model_label.load_state_dict(torch.load('/home/nahad/pycharm_mac/semi_with_cp/semi_ts_clustering/models/temp_label'))
        model.load_state_dict(torch.load('/home/nahad/pycharm_mac/semi_with_cp/semi_ts_clustering/models/temp_model'))

        for ep in range(0, 400):
            print('Labelled Epoch number: {0}'.format(ep))
            'Lambda for cp pairs'
            lam_label = 1
            # lam_u = 0.001
            lam_u = 0

            lam_cp = 1
            # torch.save(model_label.state_dict(),'/home/nahad/pycharm_mac/semi_with_cp/semi_ts_clustering/models/HCI_pairs_freehand')
            train_semisup(pair_train_btch, balanced_train_btch, lam_cp, lam_label, lam_u, ep)
            loss = evaluate_pairs(pair_data_test)
            print('Loss Semi CP rep for loop number{0}'.format(k))
            writer.add_scalar('Vlaidation Pairwise loss', loss, ep)

        for ep in range(0, 2000):
            lam_l = 1
            lam_u = 0.1  # train_feedforward_semisup(balanced_train_btch, pair_train_btch,lam_u, lam_label)

            train_feedforward(balanced_train_btch, 0)

        acc_semisup = evaluate_classification(label_data_test, 1, 1)

        semi_cp_result_list.append(acc_semisup)

        print("Done")