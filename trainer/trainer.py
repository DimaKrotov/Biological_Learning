import numpy as np
from pandas import NA
import torch
from torchvision.utils import make_grid
from models.metric import accuracy


def train(model, dataloader, optimizer, criteria, device, writer, epoch):
    running_loss = 1.0
    running_metric = 0
    N = 0
    for data in dataloader:
        inputs, label = data
        batch_size = inputs.size(0)
        inputs, label = inputs.view(batch_size, -1).to(device), label.to(device)
        # predict
        output = model(inputs)
        # loss
        loss = criteria(output, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # update running loss
        running_loss += loss.item()
        running_metric += accuracy(output, label)
        N += batch_size
    # compute stats
    L = running_loss / N
    A = running_metric / N
    writer.add_scalar("Loss", L, epoch)
    writer.add_scalar("Accuracy", A, epoch)
    print(f"Training loss: {L:.3f}, training accuracy: {A:.3f}")


def test(model, dataloader, criteria, device, writer, epoch):
    running_loss = 0.0
    running_metric = 0
    N = 0
    for data in dataloader:
        inputs, label = data
        batch_size = inputs.size(0)
        inputs, label = inputs.view(batch_size, -1).to(device), label.to(device)
        # predict
        output = model(inputs)
        # loss
        loss = criteria(output, label)
        # update running loss
        running_loss += loss.item()
        running_metric += accuracy(output, label)
        N += batch_size
    L = running_loss / N
    A = running_metric / N
    writer.add_scalar("Loss", L, epoch)
    writer.add_scalar("Accuracy", A, epoch)
    print(f"Test loss: {L:.3f}, test accuracy: {A:.3f}")