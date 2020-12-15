import torch
import numpy as np
import itertools




'''Increases the numnber of pairs per sequence. If a pair is similar, all points combinations within a segment are made
similar. 2 x N C 2 combinations. If a pair is dissimilar, all points combination of dissimilar points is taken (n^2)'''
def increase_pairs(X1,X2,Y,X1_label = None, X2_label=None):

    no_points = X1.shape[0]
    temp = list(range(0, no_points))
    if X1_label == None:
        if Y[0] == 1:
             c2_combs = list(itertools.combinations(temp, 2))
             idx1 = list(map(lambda x : x[0], c2_combs))
             idx2 = list(map(lambda x : x[1], c2_combs))
             X1_temp_1 = X1[idx1,:]
             X2_temp_1 = X1[idx2,:]
             X1_temp_2 = X2[idx1,:]
             X2_temp_2 = X2[idx2,:]
             X1 = torch.cat((X1,X1_temp_1,X1_temp_2), 0)
             X2 = torch.cat((X2,X2_temp_1,X2_temp_2), 0)
             new_l = X1.shape[0]
             Y = torch.from_numpy(np.ones((new_l,1)))
        elif Y[0] == -1:
             idx1 = list(itertools.chain.from_iterable(itertools.repeat(x, no_points) for x in temp))
             idx2 = list(itertools.chain.from_iterable(itertools.repeat(temp, no_points)))
             X1  = X1[idx1,:]
             X2 = X2[idx2,:]
             new_l = X1.shape[0]
             Y =  torch.from_numpy(- 1 * np.ones((new_l, 1)))
        return X1, X2, Y
    else:
        if Y[0] == 1:
            c2_combs = list(itertools.combinations(temp, 2))
            idx1 = list(map(lambda x: x[0], c2_combs))
            idx2 = list(map(lambda x: x[1], c2_combs))
            X1_temp_1 = X1[idx1, :]
            X2_temp_1 = X1[idx2, :]
            X1_temp_2 = X2[idx1, :]
            X2_temp_2 = X2[idx2, :]
            X1 = torch.cat((X1, X1_temp_1, X1_temp_2), 0)
            X2 = torch.cat((X2, X2_temp_1, X2_temp_2), 0)
            new_l = X1.shape[0]
            Y = torch.from_numpy(np.ones((new_l, 1)))
            X1_label = torch.from_numpy(X1_label[0].numpy() * np.ones((new_l, 1)))
            X2_label = torch.from_numpy(X2_label[0].numpy() * np.ones((new_l, 1)))
        elif Y[0] == -1:
            idx1 = list(itertools.chain.from_iterable(itertools.repeat(x, no_points) for x in temp))
            idx2 = list(itertools.chain.from_iterable(itertools.repeat(temp, no_points)))
            X1 = X1[idx1, :]
            X2 = X2[idx2, :]
            new_l = X1.shape[0]
            Y = torch.from_numpy(- 1 * np.ones((new_l, 1)))
            X1_label = torch.from_numpy(X1_label[0].numpy() * np.ones((new_l, 1)))
            X2_label = torch.from_numpy(X2_label[0].numpy() * np.ones((new_l, 1)))
        return X1, X2, Y, X1_label, X2_label



'Iterating through all segments in a batch'
def increase_all_pairs(X1,X2,Y,X1_label = None, X2_label = None):
    if X1_label == None:
        no_points = X1.shape[0]
        temp = list(range(0,no_points))
        d = [increase_pairs(X1[i,:,:], X2[i,:,:], Y[i]) for i in temp]
        X1 = list(map(lambda x : x[0], d))
        X2 = list(map(lambda x : x[1], d))
        Y = list(map(lambda x: x[2], d))
        X1 = torch.cat(X1, 0)
        X2 = torch.cat(X2, 0)
        Y = torch.cat(Y,0)
        if X1.is_cuda:
            Y = Y.cuda()
        return X1,X2,Y
    else:
        no_points = X1.shape[0]
        temp = list(range(0, no_points))
        d = [increase_pairs(X1[i, :, :], X2[i, :, :], Y[i], X1_label[i], X2_label[i]) for i in temp]
        X1 = list(map(lambda x: x[0], d))
        X2 = list(map(lambda x: x[1], d))
        Y = list(map(lambda x: x[2], d))
        X1_label = list(map(lambda x: x[3], d))
        X2_label = list(map(lambda x: x[4], d))
        X1 = torch.cat(X1, 0)
        X2 = torch.cat(X2, 0)
        Y = torch.cat(Y, 0)
        X1_label = torch.cat(X1_label,0)
        X2_label = torch.cat(X2_label,0)
        if X1.is_cuda:
            Y = Y.cuda()
        return X1, X2, Y, X1_label,X2_label


def get_f1_score(pred,Y,no_classes):
    f1_list = []
    pred =  np.array(pred)
    Y = np.array(Y)
    for i in range(0,no_classes):
        tp = sum(pred[[Y==i]] ==i)
        fp = sum(pred[[Y!=i]] == i)
        fn = sum(pred[[Y==i]] != i)
        recall = tp/(tp+fn)
        precision = tp/(tp+fp)
        f1 = (2*recall*precision)/(precision+recall)
        if np.isnan(f1) or np.isinf(f1) :
            f1 = 0


        f1_list.append(f1)
    return f1_list


'Function to get pairs of output probabilities from batch'

def PairEnum(x,mask=None):
    # Enumerate all pairs of feature in x
    assert x.ndimension() == 2, 'Input dimension must be 2'
    x1 = x.repeat(x.size(0),1)
    x2 = x.repeat(1,x.size(0)).view(-1,x.size(1))
    if mask is not None:
        xmask = mask.view(-1,1).repeat(1,x.size(1))
        #dim 0: #sample, dim 1:#feature
        x1 = x1[xmask].view(-1,x.size(1))
        x2 = x2[xmask].view(-1,x.size(1))
    return x1,x2


def PairEnum_window(x,mask=None):
    # Enumerate all pairs of feature in x
    assert x.ndimension() == 3, 'Input dimension must be 2'
    x1 = x.repeat(x.size(0),1,1)
    x2 = x.repeat(1,x.size(0),1).view(-1,x.size(1),x.size(2)) #Need to look at this
    if mask is not None:
        xmask = mask.view(-1,1).repeat(1,x.size(1))
        #dim 0: #sample, dim 1:#feature
        x1 = x1[xmask].view(-1,x.size(1))
        x2 = x2[xmask].view(-1,x.size(1))
    return x1,x2


'Function to get pariwise labels (1 sim or -1 dissim) from batch'

def Class2Simi(x,mode=None,mask=None):
    # Convert class label to pairwise similarity
    n=x.nelement()
    assert (n-x.ndimension()+1)==n,'Dimension of Label is not right'
    expand1 = x.view(-1,1).expand(n,n)  #view is to rearrange or reshape a tensor. Expand to repeat or replicate
    expand2 = x.view(1,-1).expand(n,n)
    out = expand1 - expand2
    out[out!=0] = -1 #dissimilar pair: label=-1
    out[out==0] = 1 #Similar pair: label=1
    if mode=='cls':
        out[out==-1] = 0 #dissimilar pair: label=0
    if mode=='hinge':
        out = out.float() #hingeloss require float type
    if mask is None:
        out = out.view(-1)
    else:
        mask = mask.detach()
        out = out[mask]
    return out