import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import numpy as np
import torch.nn.functional as F


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlockEncoder(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlockEncoder, self).__init__()
        self.dilation = dilation
        self.stride = stride
        self.kernel_size = kernel_size

        self.kernel_size = kernel_size

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.batch1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()

        self.dropout1 = nn.Dropout(dropout)


        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.batch2 = nn.BatchNorm1d(n_outputs)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, self.batch1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2, self.batch2)

        #self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
        #                        self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        temp = np.ones(self.kernel_size)
        temp[0: int(self.kernel_size/2)] = -1 * temp[0: int(self.kernel_size/2)]
        self.conv1.weight.data.normal_(0, 0.000001)
        self.conv1.weight.data[:,:,:] = torch.from_numpy(temp)
        self.conv2.weight.data.normal_(0, 0.000001)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalBlockDecoder(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, drop_end,dropout=0.2):
        super(TemporalBlockDecoder, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.padding = padding
        self.deconv1 = weight_norm(nn.ConvTranspose1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(drop_end)
        self.relu1 = nn.ReLU()

        self.dropout1 = nn.Dropout(dropout)
        self.batch1 = nn.BatchNorm1d(n_outputs)

        self.deconv2 = weight_norm(nn.ConvTranspose1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(drop_end)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.batch2 = nn.BatchNorm1d(n_outputs)

        self.net = nn.Sequential(self.deconv1, self.chomp1,  self.batch1, self.relu1, self.dropout1,
                                 self.deconv2, self.batch2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.ConvTranspose1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        temp = np.ones(self.kernel_size)
        temp[0: int(self.kernel_size/2)] = -1 * temp[0: int(self.kernel_size/2)]
        self.deconv1.weight.data.normal_(0, 0.000001)
        self.deconv1.weight.data[:,:,:] = torch.from_numpy(temp)
        self.deconv2.weight.data.normal_(0, 0.000001)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlockEncoder(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.layers = layers
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TcnnAutoEncoder(nn.Module):
    def __init__(self, num_inputs, output_size, num_channels, kernel_size=2, dropout=0.2):
        super(TcnnAutoEncoder, self).__init__()
        enc_layers = []
        dec_layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            enc_layers += [TemporalBlockEncoder(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.enc_layers = enc_layers
        self.enc_network = nn.Sequential(*enc_layers)
        self.linear_enc = nn.Linear(num_channels[-1], output_size)
        self.linear_dec = nn.Linear(output_size, num_channels[-1])
        self.reverse_nchannel = list( reversed(num_channels) )
        for i in range(num_levels):
            dilation_size = 2 ** (num_levels - (i+1) )
            in_channels =  self.reverse_nchannel[i]
            out_channels = self.reverse_nchannel[i+1] if i < (num_levels-1) else num_inputs
            dec_layers += [TemporalBlockDecoder(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding= 0, drop_end=(kernel_size-1) * dilation_size,dropout=dropout)]

        self.dec_layers = dec_layers
        self.dec_network = nn.Sequential(*dec_layers)

    def forward(self, x):
        encoded = self.enc_network(x)
        encoded = self.linear_enc(encoded.transpose(1, 2))
        encoded= F.normalize(encoded, p=2, dim=2);
        #encoded  = torch.mean(encoded , 1)
        #encoded = encoded.unsqueeze(1)
        decoded = self.linear_dec(encoded)
        decoded = self.dec_network(decoded.transpose(1,2))
        return  decoded, encoded