import numpy as np
import torch
from torchvision.utils import make_grid


def train(model, dataloader, optimizer, criteria, metric, device):
    running_loss = 0.0
    running_metric = 0
    for data in dataloader:
        input, label = data
        input, label = input.to(device), label.to(device)
        # predict
        output = model(input)
        # loss
        loss = criteria(output, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # update running loss
        running_loss += loss.item()
        running_metric += metric(output, label)
    return running_loss, running_metric


def test(model, dataloader, criteria, metric, device):
    running_loss = 0.0
    running_metric = 0
    for data in dataloader:
        input, label = data
        input, label = input.to(device), label.to(device)
        # predict
        output = model(input)
        # loss
        loss = criteria(output, label)
        # update running loss
        running_loss += loss.item()
        running_metric += metric(output, label)
    return running_loss, running_metric
