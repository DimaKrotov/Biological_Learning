import os, sys
import json
import pathlib
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from models.model import BPNet
from trainer.trainer import train, test
from cmd import args

# data augmentation using transforms.Compose
DSToUse = "CIFAR10"  # Or MNIST

if DSToUse == "CIFAR10":
    # CIFAR10 - (60000, 3, 32, 32)
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.5, hue=0.3),
            transforms.RandomRotation(degrees=25),
            transforms.RandomHorizontalFlip(
                p=0.5
            ),  # Random horizontal flip 50% of the time
            transforms.RandomGrayscale(
                p=0.1
            ),  # Convert to grayscale randomly 10% of the time
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,)) # MNIST stats
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # CIFAR 10 stats
        ]
    )

    batch_size = args.batch_size
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            "data", train=True, download=True, transform=transform
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            "data", train=False, download=True, transform=transform
        ),
        batch_size=batch_size,
        shuffle=True,
    )

elif DSToUse == "MNIST":
    # MNIST - (70000,1, 28,28)
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomRotation(degrees=25),
            transforms.RandomPerspective(distortion_scale=0.5, p=0.5,),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST stats
            # transforms.Normalize((0.5, 0.5), (0.5, 0.5, 0.5)),  # CIFAR 10 stats
        ]
    )

    batch_size = args.batch_size
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            "data", train=True, download=True, transform=transform
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            "data", train=False, download=True, transform=transform
        ),
        batch_size=batch_size,
        shuffle=True,
    )


# model
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# model = BPNet(28 * 28, 1, 2000, 10).to(device) # MNIST
model = BPNet(32 * 32, 3, 2000, 10).to(device)  # CIFAR 10

# training
E = args.epochs
lr = args.lr
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# record
i = 0
while True:
    run_base_dir = pathlib.Path("logs") / f"{args.sess}_try={str(i)}"
    if not run_base_dir.exists():
        os.makedirs(run_base_dir)
        break
    i += 1
with open(run_base_dir / "args.json", "w") as f:
    json.dump(vars(args), f)
train_writer = SummaryWriter(run_base_dir / "train")
test_writer = SummaryWriter(run_base_dir / "test")

# main training loop
for epoch in tqdm(range(E), desc="Epoch", total=args.epochs, dynamic_ncols=True):
    # train
    train(
        model,
        train_loader,
        optimizer,
        nn.CrossEntropyLoss(),
        device,
        train_writer,
        epoch,
    )
    # test
    test(model, test_loader, nn.CrossEntropyLoss(), device, test_writer, epoch)
    # save checkpoint
    if epoch % 10 == 9:
        torch.save(model.state_dict(), run_base_dir / f"epoch_{epoch+1}.pt")
torch.save(model.state_dict(), run_base_dir / f"epoch_{epoch+1}.pt")
