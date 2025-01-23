#!/usr/bin/env python3

import argparse
from glob import glob
from itertools import chain

from torch.utils.data import DataLoader, IterableDataset
import awkward as ak
import numpy as np
import torch
import uproot

device = "cpu"


# workaround for ancient awkward in the eic-shell
def to_torch(arr):
    if hasattr(ak, "to_torch"):
        return ak.to_torch(arr)
    else:
        return torch.Tensor(arr.to_numpy())


class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_stack = torch.nn.Sequential(
            torch.nn.Linear(3 + 1 * 80 * 116, 16),
            torch.nn.ReLU(),
            #torch.nn.Dropout(0.05),
            torch.nn.Linear(16, 2),
            torch.nn.Softmax(dim=-1),
        )

    def forward(self, x_track, x_dirc):
        # Reformat argument tensors
        x = torch.cat([ # concatenate flattened tensors
            x_track[...,0,:], # take only first track
            torch.flatten(x_dirc, -3, -1), # flatten 2d histogram of DIRC hit
        ], dim=-1)
        # Feed into the network
        return self.linear_stack(x)


class MyIterableDataset(IterableDataset):
    def __init__(self, filename_glob, batch_size=512, batches_per_step=100):
        executor = uproot.ThreadPoolExecutor(8)
        self.batch_size = batch_size
        self.batches_per_step = batches_per_step
        self.iter = uproot.iterate(
            {filename: "events" for filename in glob(filename_glob)},
            decompression_executor=executor,
            step_size=batch_size * batches_per_step,
            filter_name=[
                "_DIRCBarrelParticleIDTrackInput_features_shape",
                "_DIRCBarrelParticleIDTrackInput_features_floatData",
                "_DIRCBarrelParticleIDDIRCInput_features_shape",
                "_DIRCBarrelParticleIDDIRCInput_features_floatData",
                "_DIRCBarrelParticleIDPIDTarget_shape",
                "_DIRCBarrelParticleIDPIDTarget_int64Data",
            ],
        )

    def __iter__(self):
        for events in self.iter:
            shape = events["_DIRCBarrelParticleIDTrackInput_features_shape"][0]
            x_track = ak.unflatten(
                events["_DIRCBarrelParticleIDTrackInput_features_floatData"],
                shape[1],
                axis=-1,
            )[:,:1] # ak.to_torch() doesn't support jaggedness, so take first track only
            shape = events["_DIRCBarrelParticleIDDIRCInput_features_shape"][0]
            x_dirc = ak.unflatten(
                ak.unflatten(
                    events["_DIRCBarrelParticleIDDIRCInput_features_floatData"],
                    shape[1] * shape[2],
                    axis=-1,
                ),
                shape[2],
                axis=-1,
            )
            shape = events["_DIRCBarrelParticleIDPIDTarget_shape"][0]
            ys = ak.unflatten(
                ak.values_astype(events["_DIRCBarrelParticleIDPIDTarget_int64Data"], float),
                shape[1],
                axis=-1,
            )[:,0] # identifying the first track

            x_track = to_torch(x_track).to(device)
            x_dirc = to_torch(x_dirc).to(device)
            ys = to_torch(ys).to(device)

            num_batches = max(0, int(len(events) / self.batch_size) - 1)
            for batch_ix in range(num_batches):
                select = slice(self.batch_size * batch_ix, self.batch_size * (batch_ix + 1))
                yield x_track[select], x_dirc[select], ys[select]


def train(dataloader, model, loss_fn, optimizer):
    model.train()

    losses = []
    accs = []

    for batch, (X1, X2, y) in enumerate(dataloader):
        X1, X2, y = X1[0], X2[0], y[0]
        # Compute prediction error
        pred = model(X1, X2)
        loss = loss_fn(pred, y)
        losses.append(loss.item())
        acc = (torch.round(pred[...,0]) == y[...,0]).double().mean().item()
        accs.append(acc)
    
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return np.mean(losses), np.mean(accs)


def _eval(dataloader, model, loss_fn):
    model.eval()

    for batch, (X1, X2, y) in enumerate(dataloader):
        X1, X2, y = X1[0], X2[0], y[0]

        # Compute prediction error
        pred = model(X1, X2)
        loss = loss_fn(pred, y)
        acc = (torch.round(pred[...,0]) == y[...,0]).double().mean().item()

        break # Use only first batch
    
    return loss, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True, help="A *.hackathon.edm4eic.root file for training")
    parser.add_argument("--eval", type=str, required=True, help="A *.hackathon.edm4eic.root file for evaluation")
    args = parser.parse_args()

    # Initialize training
    model = NeuralNetwork()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(50):
        dataset = MyIterableDataset(args.train)
        dataloader = DataLoader(dataset)
        train_loss, train_acc = train(dataloader, model, loss_fn, optimizer)
        dataset = MyIterableDataset(args.eval, batch_size=10000, batches_per_step=2)
        dataloader = DataLoader(dataset)
        eval_loss, eval_acc = _eval(dataloader, model, loss_fn)
        # Metrics are on training dataset, but please implement your own
        print(f"epoch: {epoch:>7d}, train loss: {train_loss:>7f}, train accuracy: {train_acc:>7f}, eval loss: {eval_loss:>7f}, eval accuracy: {eval_acc:>7f}")
        torch.save(model.state_dict(), f"model_weights_epoch_{epoch:04d}.pth")

if __name__ == '__main__':
    main()
