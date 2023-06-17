import click
import copy
import numpy as np
import torch
import torch.utils.data as dutils
import torchhd.functional as F
from tqdm import tqdm
import pandas as pd
import os

from fedhd.client import Client, NNClient
from fedhd.data import separate_by_class, distribute_data
from pl_bolts.models.self_supervised import SimCLR


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
        nn_flag,
        niid=False,
        model=None,
        lr=3e-3,
        expt="test",
        fhdnn=-1,
        pamap_flag=False,
        imsize=32,
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
        self.niid = niid
        self.verbose = verbose
        self.embedding = embedding
        self.test_ds = test_dataset
        self.nn_flag = nn_flag
        self.fhdnn = fhdnn
        self.lr = lr
        self.model = model
        self.gpu = gpu
        self.expt = expt
        self.fhdnn = fhdnn
        self.imsize = imsize

        self.gen_client_datasets(dataset)
        self.initialize_clients(
            dim,
            nclasses,
            epochs,
            batch_size,
            gpu,
            verbose,
            pamap_flag,
            model,
            fhdnn,
            imsize,
        )

    def initialize_clients(
        self,
        dim,
        nclasses,
        epochs,
        batch_size,
        gpu,
        verbose,
        pamap_flag,
        nn=None,
        fhdnn=-1,
        imsize=32,
    ):
        """
        Initializes all clients with the provided model and also assigns it a
        partition of the input dataset.
        """

        clients = []
        if nn is not None:
            dev = gpu
            # if gpu:
            #     dev = "cuda"
            # else:
            #     dev = "cpu"

            for idx in range(self.nc):
                ds_idx = self.splits[idx]
                client_idx = NNClient(
                    nn,
                    ds_idx,
                    nclasses,
                    epochs,
                    batch_size,
                    self.lr,
                    dev,
                    verbose,
                    pamap_flag,
                )
                clients.append(client_idx)
        else:
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
                    fhdnn,
                    imsize,
                )
                clients.append(client_idx)

        assert len(clients) == self.nc
        self.clients = clients

    def gen_client_datasets(self, dataset):
        """
        Creates data splits for each client. The partition sampling is iid.
        """
        if self.niid:
            classwise = separate_by_class(dataset)
            self.splits = distribute_data(classwise, self.nc)
        else:
            dlen = len(dataset)
            split_len = dlen // self.nc
            last_split = dlen % self.nc
            split_arr = [split_len] * self.nc

            if last_split != 0:
                split_arr[-1] += last_split

            assert sum(split_arr) == dlen
            assert len(split_arr) == self.nc

            self.splits = dutils.random_split(dataset, split_arr)

        # if self.verbose:
        #     click.echo(f"Data len: {dlen} length: {split_len} sum: {sum(split_arr)}")

    def average_nns(self, trained_models):
        weights = [mdl.state_dict() for mdl in trained_models]
        master = copy.deepcopy(weights[0])

        for key in master.keys():
            for w in weights[1:]:
                master[key] += w[key]
            master[key] = torch.div(master[key], len(weights))

        return master

    def logwrite(self, round, acc):
        df = pd.DataFrame(
            {"Communication Round": [round], "Accuracy": [acc.cpu().numpy()]}
        )

        root = "/home/the-noetic/cookiejar/FedHD"
        logdir = "logs"
        logfile = os.path.join(root, logdir, f"{self.expt}.csv")

        if round == 0:
            df.to_csv(logfile, header=True, index=False)
        else:
            df.to_csv(logfile, mode="a", header=False, index=False)

    def train(self):
        """
        Simulated the federated learning process. For each round of
        communication clients are randomly chosen and each of the chosen
        client is made to train on its corresponding local data partition.
        Performs model aggregation and measures test accuracy.
        """
        num = np.ceil(self.fraction * self.nc).astype(np.intc)

        pbar = tqdm(total=self.rounds)
        tbar = tqdm(total=num, leave=False)

        if self.nn_flag:
            for round in range(self.rounds):
                pbar.set_description(f"{round}")
                choices = np.arange(0, self.nc)
                chosen = np.random.choice(choices, size=(num,), replace=True)

                trained_models = []
                tbar.reset()
                for idx, cidx in enumerate(chosen):
                    tbar.set_description(f"{idx}")
                    self.clients[cidx].train()
                    new_cmodel = self.clients[cidx].send_model()
                    trained_models.append(new_cmodel)
                    tbar.update()

                new_model = self.average_nns(trained_models)

                for client in self.clients:
                    client.update_model(new_model)

                self.model.load_state_dict(new_model)
                test_acc = self.eval(self.model)
                self.logwrite(round, test_acc)
                pbar.update()
                pbar.set_postfix({"acc": f"{test_acc}"})
            tbar.close()
            pbar.close()
            test_acc = self.eval(self.model)
            click.echo(f"final acc: {test_acc}")
        else:
            for round in range(self.rounds):
                pbar.set_description(f"{round}")
                choices = np.arange(0, self.nc)
                chosen = np.random.choice(choices, size=(num,), replace=True)
                # class_hvs_update = F.random_hv(self.nclasses, self.dim).cuda()
                class_hvs_update = torch.zeros((self.nclasses, self.dim)).to(self.gpu)

                tbar.reset()
                for idx, cidx in enumerate(chosen):
                    tbar.set_description(f"{idx}")
                    self.clients[cidx].train()
                    new_hvs = self.clients[cidx].send_model()
                    class_hvs_update = F.bundle(class_hvs_update, new_hvs)
                    tbar.update()

                class_hvs_update /= num
                self.broadcast(class_hvs_update)
                test_acc = self.eval(class_hvs_update)
                self.logwrite(round, test_acc)
                pbar.update()
                pbar.set_postfix({"acc": f"{test_acc}"})

            tbar.close()
            pbar.close()
            test_acc = self.eval(class_hvs_update)
            click.echo(f"Final accuracy: {test_acc}")

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

        dev = self.gpu
        # if self.gpu:
        #     dev = "cuda"
        #     class_hvs = class_hvs.cuda()
        # else:
        #     dev = "cpu"

        if self.nn_flag:
            model = class_hvs.to(dev)
            for batch_idx, batch in enumerate(dl):
                x, y = batch
                x = x.to(dev).float()
                y = y.to(dev)
                scores = model(x)
                _, preds = torch.max(scores, dim=-1)
                acc = [preds[idx] == y[idx] for idx in range(len(preds))]
                test_acc += sum(acc) / len(acc)
        else:
            if self.fhdnn == 2:
                # weight_path = "https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt"
                weight_path = (
                    "/home/the-noetic/cookiejar/FedHD/simclr_models/simclr2.ckpt"
                )
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
                weight_path = (
                    "/home/the-noetic/cookiejar/FedHD/simclr_models/simclr1.ckpt"
                )
                feature_extractor = SimCLR.load_from_checkpoint(
                    weight_path,
                    strict=False,
                    dataset="cifar10",
                    maxpool1=False,
                    first_conv=False,
                    input_height=self.imsize,
                ).to(dev)
                feature_extractor.freeze()

            for batch_idx, batch in enumerate(dl):
                x, y = batch
                y = y.to(dev)
                x = x.to(dev).float()

                if self.fhdnn == 1 or self.fhdnn == 2:
                    x = feature_extractor(x)

                hvs = self.embedding(x).sign()
                scores = hvs @ class_hvs.T
                _, preds = torch.max(scores, dim=-1)

                acc = [preds[idx] == y[idx] for idx in range(len(preds))]
                test_acc += sum(acc) / len(acc)

        test_acc /= batch_idx + 1

        return test_acc
