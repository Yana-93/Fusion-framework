import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import MatrixFusionLayer, ExplicitLayer

class SA_GNN(nn.Module):
    def __init__(self, nfeat, nhid, hidden_LSTM, hidden_spillover, nclass, dropout, alpha):
        super(SA_GNN, self).__init__()
        self.dropout = dropout
        self.nhid = nhid
        self.nclass = nclass
        self.alpha = alpha

        self.fusion = MatrixFusionLayer(nfeat, nhid)
        self.lstm = nn.LSTM(nhid, hidden_LSTM)
        self.explicit = ExplicitLayer(hidden_LSTM, nhid, dropout)
        self.out = nn.Linear(hidden_LSTM + hidden_spillover, nclass)

        self.graph_W = nn.Parameter(torch.empty(size=(hidden_LSTM, hidden_spillover)))
        nn.init.xavier_uniform_(self.graph_W.data, gain=1.414)

    def forward(self,A_Ind, A_News, A_Supply, inputdata):

        # Obtain the fusion features
        n_window = inputdata.size(0)
        n_firm = inputdata.size(1)
        x = torch.cat([self.fusion(day_data) for day_data in inputdata], dim=0)
        x = x.reshape(n_window, n_firm, -1)
        # lstm
        out, (h, c) = self.lstm(x)
        # Sequential Embeddings
        x = h[0]  # shape:[number of firms, the dimension of the Sequential Embedding]
        # link
        relation = self.explicit(A_Ind, A_News, A_Supply, x)
        # Softmax
        zero_mat = -9e15 * torch.ones_like(relation, device=inputdata.device)
        A = torch.where(relation > 0, relation, zero_mat)
        A = F.softmax(A, dim=1)
        A = F.dropout(A, self.dropout, training=self.training)
        # Obtain the Spillovers Embeddings
        Mh = torch.matmul(x, self.graph_W)
        H = torch.matmul(A, Mh)
        H = torch.relu(H)
        # concat: Spillovers Embeddings || Sequential Embeddings (|| denotes concatenation)
        H = torch.cat([H, x], -1)
        # output mapping
        output = self.out(H)

        return F.log_softmax(output, dim=1)
