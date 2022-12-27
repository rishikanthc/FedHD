import copy

import torch
import torch.utils.data as dutils
import torchhd.functional as F


class Client:
    def __init__(
        self, embedding, class_hvs, dataset, nclasses, epochs, batch_size, device
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
        self.device = device
        self.bs = batch_size
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

        for epoch in range(self.epochs):
            epoch_acc = 0
            for batch_idx, batch in enumerate(self.dl):
                x, y = batch
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

                acc = [preds[idx] == y[idx] for idx in range(len(y))]
                epoch_acc += sum(acc)
            print(f"epoch: {epoch} accuracy: {epoch_acc/batch_idx}")

    def update_model(self, new_class_hvs):
        self.class_hvs = copy.deepcopy(new_class_hvs)

    def send_model(self):
        return self.class_hvs