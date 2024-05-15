import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

from needle.data import MNISTDataset, DataLoader

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    # raise NotImplementedError()
    main_path = nn.Sequential(nn.Linear(dim, hidden_dim), norm(hidden_dim), nn.ReLU(), 
                              nn.Dropout(p=drop_prob), nn.Linear(hidden_dim, dim), norm(dim))
    res = nn.Residual(main_path)
    return nn.Sequential(res, nn.ReLU())
    ## END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    # raise NotImplementedError()
    resnet = nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU(), 
                           *[ResidualBlock(dim=hidden_dim, hidden_dim=hidden_dim // 2, norm=norm, drop_prob=drop_prob) for _ in range(num_blocks)], 
                           nn.Linear(hidden_dim, num_classes))
    return resnet
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    # raise NotImplementedError()
    # dataloader()
    epoch_loss = 0
    loss_fn = nn.SoftmaxLoss()
    if opt is None:
        model.eval()
        for X, y in enumerate(dataloader):
            pred = model(X)
            loss = loss_fn(pred, y)
            epoch_loss += loss
    else:
        model.train()
        for X, y in dataloader:
            pred = model(X)
            loss = loss_fn(pred, y)
            epoch_loss += loss

            loss.backward()
            opt.step()
    sample_num = len(dataloader.dataset)
    return epoch_loss/sample_num
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    # raise NotImplementedError()
    resnet = MLPResNet(dim=28*28, hidden_dim=hidden_dim, num_classes=10)
    opt = optimizer(resnet.parameters(), lr=lr, weight_decay=weight_decay)
    train_set = MNISTDataset(image_filename = os.path.join(data_dir, 'train-images-idx3-ubyte.gz'), 
                             label_filename=os.path.join(data_dir, 'train-labels-idx1-ubyte.gz'))
    test_set = MNISTDataset(image_filename = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz'), 
                             label_filename=os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz'))
    train_loder = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loder = DataLoader(test_set, batch_size=batch_size)
    for _ in range(epochs):
        train_loss = epoch(train_loder, resnet, opt)
        test_loss = epoch(test_loder, resnet)
    return train_loss, test_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
