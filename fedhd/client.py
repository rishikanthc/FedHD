import click
import torch
import torch.utils.data as dutils
import torch.nn as nn
import torchhd.functional as F
import numpy as np
import copy


class NNClient:
    def __init__(self, model, dataset, nclasses, epochs, batch_size, lr, gpu, verbose):
        """
        Initialize the client with a model and dataset. Also specifies
        parameters for local learning on the client.
        """

        self.model = model
        self.ds = dataset
        self.nc = nclasses
        self.epochs = epochs
        self.device = gpu
        self.bs = batch_size
        self.verbose = verbose
        self.lr = lr
        self.optim = torch.optim.Adam(
            self.model.parameters(), weight_decay=0.0, lr=self.lr
        )
        self.loss = nn.CrossEntropyLoss()

        self.create_dl()

    def create_dl(self):
        """
        Create Torch DataLoader from dataset
        """

        self.dl = dutils.DataLoader(
            self.ds, batch_size=self.bs, shuffle=True, num_workers=8
        )

    def train(self):
        self.model = self.model.cuda()
        for epoch in range(self.epochs):
            epoch_acc = []
            for bidx, batch in enumerate(self.dl):
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device).type(torch.long)

                self.optim.zero_grad()
                y_hat = self.model(x)
                loss = self.loss(y_hat, y)
                loss.backward()
                self.optim.step()

                _, preds = torch.max(y_hat, dim=-1)
                y = y.cpu()
                preds = preds.cpu()
                acc = [preds[idx] == y[idx] for idx in range(len(preds))]
                acc = np.mean(acc)
                epoch_acc.append(acc)

            # if self.verbose:
            #     click.echo(f"\tepoch: {epoch} accuracy: {epoch_acc}")

        self.model = self.model.cpu()

    def update_model(self, new_model):
        self.model.load_state_dict(new_model)

    def send_model(self):
        return copy.deepcopy(self.model.cpu())


class Client:
    def __init__(
        self, embedding, class_hvs, dataset, nclasses, epochs, batch_size, gpu, verbose
    ):
        """
        Initialize the client with a model and dataset. Also specifies
        parameters for local learning on the client.
        """

        self.embedding = embedding
        self.class_hvs = class_hvs  # c x D
        self.ds = dataset
        self.nc = nclasses
        self.epochs = epochs
        self.gpu = gpu
        self.bs = batch_size
        self.verbose = verbose
        self.create_dl()

    def create_dl(self):
        """
        Create Torch DataLoader from dataset
        """

        self.dl = dutils.DataLoader(
            self.ds, batch_size=self.bs, shuffle=True, num_workers=8
        )

    def train(self):
        """
        Trains the client model on local dataset for corresponding
        # of epochs specified.
        """

        if self.gpu:
            dev = 'cuda'
            self.embedding = self.embedding.cuda()
            self.class_hvs = self.class_hvs.cuda()
        else:
            dev = 'cpu'

        for epoch in range(self.epochs):
            epoch_acc = 0
            for batch_idx, batch in enumerate(self.dl):
                x, y = batch
                x = x.to(dev).float()
                y = y.to(dev)

                if self.verbose:
                    click.echo(f"\tDEBUG: data {x.dtype}")

                hvs = self.embedding(x).sign()  # bs x D
                scores = hvs @ self.class_hvs.T  # bs x c
                _, preds = torch.max(scores, dim=-1)

                for label in range(self.nc):
                    hv_label = self.class_hvs[label]
                    incorrect = hvs[torch.bitwise_and(y != preds, y == label)]
                    # incorrect = incorrect.sum(dim=0, keepdim=True).squeeze()
                    incorrect = incorrect.multibundle()

                    # self.class_hvs[label] += incorrect
                    self.class_hvs[label] = F.bundle(hv_label, incorrect)

                    incorrect = hvs[torch.bitwise_and(y != preds, preds == label)]
                    # incorrect = incorrect.sum(dim=0, keepdim=True).squeeze()
                    incorrect = incorrect.multibundle()

                    self.class_hvs[label] -= incorrect
                    self.class_hvs[label] = F.bundle(hv_label, incorrect.negative())

                acc = [
                    preds[idx].detach().cpu() == y[idx].detach().cpu()
                    for idx in range(len(y))
                ]
                epoch_acc += sum(acc) / len(acc)

            if self.verbose:
                click.echo(f"\tepoch: {epoch} accuracy: {epoch_acc/(batch_idx + 1)}")

    def update_model(self, new_class_hvs):
        self.class_hvs = new_class_hvs

    def send_model(self):
        return self.class_hvs
