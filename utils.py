import numpy as np
import torch
import pickle
from sklearn.metrics import roc_auc_score

def load_pkl(filename):
    with open(filename, "rb") as fi:
        res = pickle.load(fi)
    return res

# The sliding window size equals to 30.
def split_data(data):
    res_text = []
    for i in range(30, 730):
        res_temp = []
        for j in range(i - 30, i):
            res_temp.append(data[j])
        res_text.append(res_temp)
    return res_text

def load_data():

    x_numerical = load_pkl("./data_SP500/x_numerical.pkl")
    x_textual = load_pkl("./data_SP500/x_textual.pkl")
    y_ = load_pkl("./data_SP500/y.pkl")
    y_ = torch.tensor(y_ > 0).to(torch.long)

    industry_relation = load_pkl("./data_SP500/industry_relation.pkl")
    news_relation = load_pkl("./data_SP500/news_relation.pkl")
    supply_relation = load_pkl("./data_SP500/supply_relation.pkl")

    x_numerical_30 = split_data(x_numerical)
    x_textual_30 = split_data(x_textual)
    x = np.concatenate([x_numerical_30, x_textual_30], axis=-1)

    x_train = x[:-140]
    x_val = x[-140:-70]
    x_test = x[-70:]

    train_labels = y_[:-140]
    val_labels = y_[-140:-70]
    test_labels = y_[-70:]

    train_x = torch.tensor(x_train, dtype=torch.double)
    val_x = torch.tensor(x_val, dtype=torch.double)
    test_x = torch.tensor(x_test, dtype=torch.double)

    A_Ind = torch.tensor(industry_relation, dtype=torch.double)
    A_news = torch.tensor(news_relation, dtype=torch.double)
    A_supply = torch.tensor(supply_relation, dtype=torch.double)

    return train_x, val_x, test_x, train_labels, val_labels, test_labels, A_Ind, A_news, A_supply


def metrics(preds, labels):

    acc = sum(preds.argmax(-1) == labels) / len(labels)
    auc = roc_auc_score(labels, preds[:, 1])
    return acc, auc
