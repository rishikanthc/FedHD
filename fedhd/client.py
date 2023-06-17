import click
import torch
import torch.utils.data as dutils
from torchhd.datasets import PAMAP
import torch.nn as nn
import torchhd.functional as F
import numpy as np
import copy
from pl_bolts.models.self_supervised import SimCLR


class NNClient:
    def __init__(
        self, model, dataset, nclasses, epochs, batch_size, lr, gpu, verbose, pamap_flag
    ):
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
        self.pamap_flag = pamap_flag

        self.create_dl()

    def create_dl(self):
        """
        Create Torch DataLoader from dataset
        """

        self.dl = dutils.DataLoader(
            self.ds, batch_size=self.bs, shuffle=True, num_workers=8
        )

    def train(self):
        # self.model = self.model.cuda()
        self.model = self.model.to(self.device)
        for epoch in range(self.epochs):
            epoch_acc = []
            for bidx, batch in enumerate(self.dl):
                x, y = batch
                x = x.to(self.device).type(torch.float)
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
        self,
        embedding,
        class_hvs,
        dataset,
        nclasses,
        epochs,
        batch_size,
        gpu,
        verbose,
        fhdnn,
        imsize,
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
        self.fhdnn = fhdnn
        self.imsize = imsize
        self.create_dl()

    def create_dl(self):
        """
        Create Torch DataLoader from dataset
        """

        self.dl = dutils.DataLoader(
            self.ds, batch_size=self.bs, shuffle=True, num_workers=8
        )

    def train(self, lr=1):
        """
        Trains the client model on local dataset for corresponding
        # of epochs specified.
        """

        # if self.gpu:
        #     dev = 'cuda'
        #     self.embedding = self.embedding.cuda()
        #     self.class_hvs = self.class_hvs.cuda()
        # else:
        #     dev = 'cpu'
        dev = self.gpu
        self.embedding = self.embedding.to(dev)
        self.class_hvs = self.class_hvs.to(dev)

        if self.fhdnn == 2:
            # weight_path = "https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt"
            weight_path = "/home/the-noetic/cookiejar/FedHD/simclr_models/simclr2.ckpt"
            feature_extractor = SimCLR.load_from_checkpoint(
                weight_path,
                strict=False,
                dataset="imagenet",
                maxpool1=False,
                first_conv=True,
                input_height=self.imsize,
            ).to(dev)
            feature_extractor.freeze()
        elif self.fhdnn == 1:
            # weight_path = "https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/simclr-cifar10-v1-exp12_87_52/epoch%3D960.ckpt"
            weight_path = "/home/the-noetic/cookiejar/FedHD/simclr_models/simclr1.ckpt"
            feature_extractor = SimCLR.load_from_checkpoint(
                weight_path,
                strict=False,
                dataset="cifar10",
                maxpool1=False,
                first_conv=False,
                input_height=self.imsize,
            ).to(dev)
            feature_extractor.freeze()

        for epoch in range(self.epochs):
            epoch_acc = 0
            for batch_idx, batch in enumerate(self.dl):
                x, y = batch
                x = x.to(dev).float()
                y = y.to(dev)

                if self.verbose:
                    click.echo(f"\tDEBUG: data {x.dtype}")

                if self.fhdnn == 1 or self.fhdnn == 2:
                    x = feature_extractor(x)

                hvs = self.embedding(x).sign()  # bs x D
                scores = hvs @ self.class_hvs.T  # bs x c
                _, preds = torch.max(scores, dim=-1)

                for label in range(self.nc):
                    hv_label = self.class_hvs[label]
                    incorrect = hvs[torch.bitwise_and(y != preds, y == label)]
                    # incorrect = incorrect.sum(dim=0, keepdim=True).squeeze()
                    incorrect = incorrect.multibundle()

                    self.class_hvs[label] += incorrect * lr
                    # self.class_hvs[label] = F.bundle(hv_label, incorrect)

                    incorrect = hvs[torch.bitwise_and(y != preds, preds == label)]
                    # incorrect = incorrect.sum(dim=0, keepdim=True).squeeze()
                    incorrect = incorrect.multibundle()

                    self.class_hvs[label] -= incorrect * lr
                    # self.class_hvs[label] = F.bundle(hv_label, incorrect.negative())

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
