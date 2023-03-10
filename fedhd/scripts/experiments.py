import click
import torch.nn as nn
import torchhd as hd
import torchvision.transforms as tf
from torchvision.datasets import MNIST

from fedhd.fl_trainer import Trainer
from fedhd.ucihar import UCIHAR


class Params(object):
    def __init__(self):
        self.dim = None
        self.batch_size = None
        self.nclients = None
        self.fraction = None
        self.epochs = None
        self.gpu = None
        self.rounds = None
        self.verbose = None


pass_params = click.make_pass_decorator(Params, ensure=True)


@click.group()
@click.option(
    "-D", "--dim", default=2000, show_default=True, help="Hypervector dimensionality"
)
@click.option(
    "-bs",
    "--batch_size",
    default=5,
    show_default=True,
    help="Batch size for client training",
)
@click.option("-nc", "--nclients", default=100, show_default=True, help="# of clients")
@click.option(
    "-C",
    "--fraction",
    default=0.3,
    show_default=True,
    help="Fraction of clients to use during each round of communication",
)
@click.option(
    "-E",
    "--epochs",
    default=2,
    show_default=True,
    help="Number of epochs to train on the local client",
)
@click.option(
    "-r",
    "--rounds",
    default=50,
    show_default=True,
    help="# of rounds of federated learning",
)
@click.option("-g", "--gpu", is_flag=True, help="Enable GPU acceleration")
@click.option("-v", "--verbose", is_flag=True, help="Verbose mode")
@pass_params
def main(params, dim, batch_size, nclients, fraction, epochs, rounds, gpu, verbose):
    click.echo("Running FedHD")
    params.dim = dim
    params.batch_size = batch_size
    params.nclients = nclients
    params.fraction = fraction
    params.epochs = epochs
    params.rounds = rounds
    params.gpu = gpu
    params.verbose = verbose


@main.command()
@click.option(
    "-r",
    "--root",
    type=click.Path(exists=True, dir_okay=True),
    default="/home/paperspace/codezone/data",
    help="Root directory of the data",
)
@pass_params
def isolet(params, root):

    click.echo("Loading ISOLET dataset")
    ds = hd.datasets.ISOLET(root, train=True, download=True)
    test_ds = hd.datasets.ISOLET(root, train=False, download=True)

    click.echo("Creating HD embedding model")
    embedding = hd.embeddings.Projection(617, params.dim)

    trainer = Trainer(
        embedding,
        params.dim,
        ds,
        test_ds,
        26,
        params.batch_size,
        params.nclients,
        params.fraction,
        params.rounds,
        params.epochs,
        params.gpu,
        params.verbose,
    )
    click.echo("Running federated learning on the ISOLET Dataset")
    trainer.train()


@main.command()
@click.option(
    "-r",
    "--root",
    type=click.Path(exists=True, dir_okay=True),
    default="/home/paperspace/codezone/data",
    help="Root directory of the data",
)
@pass_params
def ucihar(params, root):

    click.echo("Loading UCIHAR dataset")
    ds = UCIHAR(root, train=True, download=False)
    test_ds = UCIHAR(root, train=False, download=False)

    click.echo("Creating HD Embedding model")
    feat_size = ds[0][0].shape[-1]
    embedding = hd.embeddings.Projection(feat_size, params.dim)

    trainer = Trainer(
        embedding,
        params.dim,
        ds,
        test_ds,
        6,
        params.batch_size,
        params.nclients,
        params.fraction,
        params.rounds,
        params.epochs,
        params.gpu,
        params.verbose,
    )
    click.echo("Running federated learning on the UCIHAR Dataset")
    trainer.train()


@main.command()
@click.option(
    "-r",
    "--root",
    type=click.Path(exists=True, dir_okay=True),
    default="/home/paperspace/codezone/data",
    help="Root directory of the data",
)
@pass_params
def pamap(params, root):

    click.echo("Loading PAMAP dataset")
    ds = hd.datasets.PAMAP(root, subjects=[1, 2, 3, 4, 5, 6], download=False)
    test_ds = hd.datasets.PAMAP(root, subjects=[7, 8], download=False)

    click.echo("Creating HD embedding model")
    feat_size = ds[0][0].shape[-1]
    embedding = hd.embeddings.Projection(feat_size, params.dim)

    trainer = Trainer(
        embedding,
        params.dim,
        ds,
        test_ds,
        18,
        params.batch_size,
        params.nclients,
        params.fraction,
        params.rounds,
        params.epochs,
        params.gpu,
        params.verbose,
    )
    click.echo("Running federated learning on the PAMAP Dataset")
    trainer.train()


@main.command()
@click.option(
    "-r",
    "--root",
    type=click.Path(exists=True, dir_okay=True),
    default="/home/paperspace/codezone/data",
    help="Root directory of the data",
)
@pass_params
def mnist(params, root):

    click.echo("Loading MNIST dataset")
    transforms = tf.Compose([tf.ToTensor()])
    ds = MNIST(root, train=True, download=True, transform=transforms)
    test_ds = MNIST(root, train=False, download=True, transform=transforms)

    click.echo("Creating HD embedding model")
    feat_size = 784
    embedding = nn.Sequential(
        nn.Flatten(), hd.embeddings.Projection(feat_size, params.dim)
    )

    trainer = Trainer(
        embedding,
        params.dim,
        ds,
        test_ds,
        18,
        params.batch_size,
        params.nclients,
        params.fraction,
        params.rounds,
        params.epochs,
        params.gpu,
        params.verbose,
    )
    click.echo("Running federated learning on the MNIST Dataset")
    trainer.train()
