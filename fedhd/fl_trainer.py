import click
import numpy as np
import torch
import torch.utils.data as dutils
import torchhd.functional as F

from fedhd.client import Client


class Trainer:
    def __init__(
        self,
        embedding,
        dim,
        dataset,
        test_dataset,
        nclasses,
        batch_size,
        nclients,
        fraction,
        rounds,
        epochs,
        gpu,
        verbose,
    ):
        """
        This class initializes the overall federated learning pipeline.
        It takes as input the starting model, the dataset and FL parameters.
        Based on the params clients are initialized and the end-to-end FL process
        is simlated
        """

        self.nc = nclients
        self.rounds = rounds
        self.nclasses = nclasses
        self.fraction = fraction
        self.dim = dim
        self.verbose = verbose
        self.embedding = embedding
        self.test_ds = test_dataset

        self.gen_client_datasets(dataset)
        self.initialize_clients(dim, nclasses, epochs, batch_size, gpu, verbose)

    def initialize_clients(self, dim, nclasses, epochs, batch_size, gpu, verbose):
        """
        Initializes all clients with the provided model and also assigns it a
        partition of the input dataset.
        """
        clients = []
        class_hvs = F.random_hv(nclasses, dim)

        for idx in range(self.nc):
            ds_idx = self.splits[idx]
            client_idx = Client(
                self.embedding,
                class_hvs,
                ds_idx,
                nclasses,
                epochs,
                batch_size,
                gpu,
                verbose,
            )
            clients.append(client_idx)

        assert len(clients) == self.nc
        self.clients = clients

    def gen_client_datasets(self, dataset):
        """
        Creates data splits for each client. The partition sampling is iid.
        """
        dlen = len(dataset)
        split_len = dlen // self.nc
        last_split = dlen % self.nc
        split_arr = [split_len] * self.nc

        if last_split != 0:
            split_arr[-1] += last_split

        assert sum(split_arr) == dlen
        assert len(split_arr) == self.nc

        if self.verbose:
            click.echo(f"Data len: {dlen} length: {split_len} sum: {sum(split_arr)}")

        self.splits = dutils.random_split(dataset, split_arr)

    def train(self):
        """
        Simulated the federated learning process. For each round of
        communication clients are randomly chosen and each of the chosen
        client is made to train on its corresponding local data partition.
        Performs model aggregation and measures test accuracy.
        """
        for round in range(self.rounds):
            num = np.ceil(self.fraction * self.nc).astype(np.intc)
            choices = np.arange(0, self.nc)
            chosen = np.random.choice(choices, size=(num,), replace=True)
            class_hvs_update = F.random_hv(self.nclasses, self.dim)

            for cidx in chosen:
                self.clients[cidx].train()
                new_hvs = self.clients[cidx].send_model()
                class_hvs_update = F.bundle(class_hvs_update, new_hvs)

            class_hvs_update /= num
            self.broadcast(class_hvs_update)
            test_acc = self.eval(class_hvs_update)

            click.echo(f"Comm round: {round} accuracy: {test_acc}")

    def broadcast(self, class_hvs_update):
        """
        Broadcasts the new updated model to all the clients after each
        round of communication
        """
        for idx in range(self.nc):
            self.clients[idx].update_model(class_hvs_update)

    def eval(self, class_hvs):
        dl = dutils.DataLoader(self.test_ds, batch_size=128, shuffle=False)
        test_acc = 0

        for batch_idx, batch in enumerate(dl):
            x, y = batch
            hvs = self.embedding(x).sign()
            scores = hvs @ class_hvs.T
            _, preds = torch.max(scores, dim=-1)

            acc = [preds[idx] == y[idx] for idx in range(len(preds))]
            test_acc += sum(acc) / len(acc)

        test_acc /= batch_idx + 1

        return test_acc
