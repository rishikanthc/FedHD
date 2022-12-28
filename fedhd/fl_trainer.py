import click
import torch.utils.data as dutils


class Trainer:
    def __init__(
        self,
        embedding,
        dataset,
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
        self.verbose = verbose
        self.splits = self.gen_client_datasets(dataset)

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
            click.echo(
                f"Data length: {dlen} split length: {split_len} split arr: {len(split_arr)} splitsum: {sum(split_arr)}"  # noqa
            )

        splits = dutils.random_split(dataset, split_arr)

        if self.verbose:
            click.echo(f"Created {len(splits)} splits")

        return splits

    def train(self):
        """
        Simulated the federated learning process. For each round of
        communication clients are randomly chosen and each of the chosen
        client is made to train on its corresponding local data partition.
        Performs model aggregation and measures test accuracy.
        """
        pass

    def broadcast(self):
        """
        Broadcasts the new updated model to all the clients after each
        round of communication
        """
