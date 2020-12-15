
import torch.nn as nn


class RNNSeriesEncoding(nn.Module):
    def __init__(self, enc_dim, data_dim):
        super(RNNSeriesEncoding, self).__init__()
        self.var_dim = data_dim
        self.RNN_hid_dim = enc_dim

        self.rnn_enc_layer = nn.GRU(self.var_dim, self.RNN_hid_dim, batch_first=True)
        self.rnn_dec_layer = nn.GRU(self.RNN_hid_dim, self.var_dim, batch_first=True)

    def forward(self, X):
        X_enc, _ = self.rnn_enc_layer(X)
        X_dec, _ = self.rnn_dec_layer(X_enc)
        return X_enc, X_dec
