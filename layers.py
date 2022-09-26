import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MatrixFusionLayer(nn.Module):
    """Matrix-based Fusion Method"""
    def __init__(self, in_features, out_features):
        super(MatrixFusionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.U = nn.Parameter(torch.empty(size=(in_features, 16)))
        nn.init.xavier_uniform_(self.U.data, gain=1.414)

        self.fc = nn.Linear(in_features + 16, out_features)


    def forward(self, inputdata):
        summed_features_emb_square = torch.matmul(inputdata, self.U) ** 2
        squared_sum_features_emb = torch.matmul(inputdata ** 2, self.U ** 2)
        x_inter = 0.5 * (summed_features_emb_square - squared_sum_features_emb)
        x_concat = torch.cat((inputdata, x_inter), dim=1)
        x = torch.tanh(self.fc(x_concat))
        return x


class Self_attention(nn.Module):
    """
    To incorporate the global influence
    """
    def __init__(self, in_features, out_features, dropout):
        super(Self_attention, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features

        self.W_Q = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W_Q.data, gain=1.414)
        self.W_K = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W_K.data, gain=1.414)
        self.W_V = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W_V.data, gain=1.414)

    def forward(self, inputdata):
        """
        inputdata: Sequential Embeddings
            shape: [198, in_features] -> (number of firms, the dimension of the input features)
        return: the global market information of each firm
            shape: [198, out_features] -> (number of firms, the dimension of the output features )
        """
        Q = torch.matmul(inputdata, self.W_Q)
        K = torch.matmul(inputdata, self.W_K)
        V = torch.matmul(inputdata, self.W_V)

        Scores = torch.matmul(Q, K.T) / math.sqrt(Q.size(-1))
        Scores_softmax = F.softmax(Scores, dim=-1)
        Scores_softmax = F.dropout(Scores_softmax, self.dropout, training=self.training)
        Market_Signals = torch.matmul(Scores_softmax, V)
        return Market_Signals

class Relation_Attention(nn.Module):
    """
    calculating the importance of each pre-defined relation.
    """
    def __init__(self, in_features, out_features, dropout):
        super(Relation_Attention, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)


    def forward(self, Adj, inputdata):
        """
        inputdata: global market information of each firm
            shape: [198, in_features] -> (number of firms, the dimension of the input features)
        Adj: original adjacency matrix
            shape: [198, 198] -> (number of firms, number of firms)
        return: adjacency matrix under the gate mechanism
            shape: [198, 198] -> (number of firms, number of firms,)
        """
        attention_input = torch.matmul(inputdata, self.W)
        attention_input = F.dropout(attention_input, self.dropout, training=self.training)
        attention = torch.sigmoid(torch.matmul(attention_input, attention_input.T))

        return torch.mul(Adj, attention)

class ExplicitLayer(nn.Module):
    """
    To combine the varying contributions of multi-relations to form the explicit link.
    """
    def __init__(self, in_features, out_features, dropout):
        super(ExplicitLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features

        self.self_attention = Self_attention(in_features, out_features, dropout)
        self.relation_attention_ind = Relation_Attention(out_features, out_features, dropout)
        self.relation_attention_news = Relation_Attention(out_features, out_features, dropout)
        self.relation_attention_supply = Relation_Attention(out_features, out_features, dropout)

    def forward(self, A_Ind, A_News, A_Supply, inputdata):

        # Obtain the global market information
        market_Signals = self.self_attention(inputdata)  # shape:[number of firms, the dimension of the global market information]
        # Obtain the adjacency matrix of News-coexposure Relation under the Attention mechanism
        A_news = self.relation_attention_news(A_News, market_Signals)
        # Obtain the adjacency matrix of Industry Relation under the Attention mechanism
        A_ind = self.relation_attention_ind(A_Ind, market_Signals)
        # Obtain the adjacency matrix of Supply Chain  Relation under the Attention mechanism
        A_supply = self.relation_attention_supply(A_Supply, market_Signals)

        # Dynamically weighting multi-relations
        A = A_news + A_ind + A_supply  # shape[number of firms, number of firms]

        return A
