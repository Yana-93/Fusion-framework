import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils import load_data, metrics
from models import SA_GNN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=9, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=200, help='Patience')
parser.add_argument('--gpu_ids', type=list, default=[0], help='Disables CUDA training.')
parser.add_argument('--accumulation_steps', type=int, default=32, help='Gradient Accumulation.')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--hidden_LSTM', type=int, default=32, help='Hidden size of LSTM.')
parser.add_argument('--hidden_spillover', type=int, default=32, help='Hidden size of spillover embedding.')
parser.add_argument('--nclass', type=int, default=2, help='Number of class.')

args = parser.parse_args([])
args.cuda = not args.no_cuda and torch.cuda.is_available()
# Load data
# Sequential data
"""
train_x, val_x, test_x: (Economic Indicators Data || Textual Media Data)  
train_x -> shape: [560, 30, 198, 13],
val_x   -> shape: [70, 30, 198, 13],
test_x  -> shape: [70, 30, 198, 13],

shape: [No. of samples, Window size, No. of firms, dimension of features],
"""
# Relation data
"""
A_Ind: adjacency matrix of the Industry Relation, shape: [198, 198]
A_news: adjacency matrix of the News-coexposure Relation, shape: [198, 198]
A_supply: adjacency matrix of the Supply Chain Relation, shape: [198, 198]

shape: [No. of firms, No. of firms]
"""

# label data
"""
train_labels, val_labels, test_labels: 
This is a binary classification problem. 
If the intraday closing price is higher than the opening price, the sample is labeled with 1, otherwise labeled with 0.
train_labels -> shape: [560, 198], 
val_labels -> shape: [70, 198], 
test_labels -> shape: [70, 198]

shape: [No. of samples, No. of firms]
"""

train_x, val_x, test_x, train_labels, val_labels, test_labels, A_Ind, A_news, A_supply = load_data()

def train(epoch):
    t = time.time()
    loss_train_total = 0
    count_train = 0
    sample_seq = list(range(train_x.size(0)))
    random.shuffle(sample_seq)
    model.train()
    for i in sample_seq:
        output = model(A_Ind, A_news, A_supply, train_x[i])
        loss = loss_fun(output, train_labels[i])
        loss_train_total += loss.item()
        count_train += 1
        loss = loss / args.accumulation_steps
        loss.backward()
        if (count_train % args.accumulation_steps) == 0:
            optimizer.step()
            optimizer.zero_grad()
    if (count_train % args.accumulation_steps) != 0:
        optimizer.step()
        optimizer.zero_grad()

    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train_total / count_train), end=' ')

    eval_result = compute_val()

    print('time: {:.4f}s'.format(time.time() - t))

    return eval_result

def compute_val():
    model.eval()
    loss_val_total = 0
    phase_pred = None
    phase_label = None
    for i in range(len(val_x)):
        output = model(A_Ind, A_news, A_supply, val_x[i])
        loss = loss_fun(output, val_labels[i])
        loss_val_total += loss.item()
        output = np.exp(output.detach().cpu().numpy())
        if phase_pred is None:
            phase_pred = output
            phase_label = val_labels[i].detach().cpu().numpy()
        else:
            phase_pred = np.concatenate((phase_pred, output), axis=0)
            phase_label = np.concatenate((phase_label, val_labels[i].detach().cpu().numpy()), axis=0)
    acc, auc = metrics(phase_pred, phase_label)
    print('loss_val: {:.4f}'.format(loss_val_total / len(val_x)),
          'acc_val: {:.4f}'.format(acc),
          'auc_val: {:.4f}'.format(auc), end=' ')
    return auc


def compute_test():
    model.eval()
    phase_pred = None
    phase_label = None
    for i in range(len(test_x)):
        output = model(A_Ind, A_news, A_supply, test_x[i])
        output = np.exp(output.detach().cpu().numpy())
        if phase_pred is None:
            phase_pred = output
            phase_label = test_labels[i].detach().cpu().numpy()
        else:
            phase_pred = np.concatenate((phase_pred, output), axis=0)
            phase_label = np.concatenate((phase_label, test_labels[i].detach().cpu().numpy()), axis=0)

    acc, auc = metrics(phase_pred, phase_label)
    print('acc_test: {:.4f}'.format(acc), end=' ')
    print('auc_test: {:.4f}'.format(auc), end=' ')

    return acc, auc




random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Model and optimizer
nfeat = train_x.size(-1)
model = SA_GNN(nfeat, args.hidden, args.hidden_LSTM, args.hidden_spillover, args.nclass, args.dropout, args.alpha)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr,
                       weight_decay=args.weight_decay)
loss_fun = nn.NLLLoss()
if (len(args.gpu_ids) == 0) or (not args.cuda):
    device = torch.device("cpu")
    print("Train Mode : CPU")
elif args.cuda and len(args.gpu_ids) > 1:
    # len(gpu_ids) > 1
    device = torch.device("cuda:0")
    model = nn.DataParallel(model, device_ids=args.gpu_ids)
    print("Train Mode : Multi GPU;", args.gpu_ids)
else:
    # len(gpu_ids) = 1
    device = torch.device("cuda:" + str(args.gpu_ids[0]) if args.cuda else "cpu")
    print("Train Mode : One GPU;", device)
model.to(torch.double)
model = model.to(device)
train_x = train_x.to(device)
val_x = val_x.to(device)
test_x = test_x.to(device)
val_labels = val_labels.to(device)
train_labels = train_labels.to(device)
test_labels = test_labels.to(device)
A_Ind = A_Ind.to(device)
A_news = A_news.to(device)
A_supply = A_supply.to(device)
print("\n", "##" * 10, "  NetWork  ", "##" * 10, "\n", model, "\n", "##" * 26, "\n")
# Training model
t_total = time.time()

auc_values = []
bad_counter = 0

best = 0
best_epoch = 0
for epoch in range(args.epochs):
    auc_values.append(train(epoch))
    torch.save(model.state_dict(), '{}.pkl'.format(epoch))
    if auc_values[-1] > best:
        best = auc_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb < best_epoch:
            os.remove(file)
files = glob.glob('*.pkl')

for file in files:
    epoch_nb = int(file.split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)


print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
# Restore the best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))
print("The seed is {}".format(args.seed))

# Testing
compute_test()


