import click
import torch.nn as nn
import torchhd as hd
import torchvision.transforms as tf
from torchvision.datasets import MNIST
from torch.utils.data import random_split

from fedhd.fl_trainer import Trainer
from fedhd.data import FaceDataset, PAMAP2Dataset

# from torchhd.datasets import UCIHAR

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
        self.nn_flag = None
        self.expt = None


pass_params = click.make_pass_decorator(Params, ensure=True)


@click.group()
@click.option(
    "-nn", "--nn_flag", is_flag=True, default=False, show_default=True, help="use nn"
)
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
@click.option("-g", "--gpu", default="cuda:0", help="Enable GPU acceleration")
@click.option("-v", "--verbose", is_flag=True, help="Verbose mode")
@click.option("-exp", "--expt", default="test", help="Exp name")
@pass_params
def main(
    params,
    dim,
    batch_size,
    nclients,
    fraction,
    epochs,
    rounds,
    gpu,
    verbose,
    nn_flag,
    expt,
):
    click.echo("Running FedHD")
    params.dim = dim
    params.batch_size = batch_size
    params.nclients = nclients
    params.fraction = fraction
    params.epochs = epochs
    params.rounds = rounds
    params.gpu = gpu
    params.verbose = verbose
    params.nn_flag = nn_flag
    params.expt = expt


@main.command()
@click.option(
    "-r",
    "--root",
    type=click.Path(exists=True, dir_okay=True),
    default="/home/the-noetic/cookiejar/data",
    help="Root directory of the data",
)
@pass_params
def isolet(params, root):

    click.echo("Loading ISOLET dataset")
    ds = hd.datasets.ISOLET(root, train=True, download=True)
    test_ds = hd.datasets.ISOLET(root, train=False, download=True)

    if params.nn_flag:
        model = nn.Sequential(
            nn.Flatten(), nn.Linear(ds[0][0].shape[-1], 128), nn.Linear(128, 26)
        )

        trainer = Trainer(
            None,
            0,
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
            params.nn_flag,
            model,
            params.expt,
        )
        click.echo("running fl on MNIST NN")
        trainer.train()
    else:
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
            params.nn_flag,
            expt=params.expt,
        )
        click.echo("Running federated learning on the ISOLET Dataset")
        trainer.train()


@main.command()
@click.option(
    "-r",
    "--root",
    type=click.Path(exists=True, dir_okay=True),
    default="/home/the-noetic/cookiejar/data",
    help="Root directory of the data",
)
@pass_params
def ucihar(params, root):

    click.echo("Loading UCIHAR dataset")
    ds = UCIHAR(root, train=True, download=False)
    test_ds = UCIHAR(root, train=False, download=False)

    if params.nn_flag:
        model = nn.Sequential(
            nn.Flatten(), nn.Linear(ds[0][0].shape[-1], 128), nn.Linear(128, 6)
        )

        trainer = Trainer(
            None,
            0,
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
            params.nn_flag,
            model,
            params.expt,
        )
        click.echo("running fl on MNIST NN")
        trainer.train()
    else:
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
            params.nn_flag,
            expt=params.expt,
        )
        click.echo("Running federated learning on the UCIHAR Dataset")
        trainer.train()


@main.command()
@click.option(
    "-r",
    "--root",
    type=click.Path(exists=True, dir_okay=True),
    default="/home/the-noetic/cookiejar/data",
    help="Root directory of the data",
)
@pass_params
def pamap(params, root):

    click.echo("Loading PAMAP dataset")
    ds = hd.datasets.PAMAP(root, subjects=[1, 7, 8, 4, 5, 6], download=False)
    test_ds = hd.datasets.PAMAP(root, subjects=[2, 3], download=False)
    # all_ds = PAMAP2Dataset(root)
    # split = int(0.75 * len(all_ds))
    # rem = len(all_ds) - split
    # ds, test_ds = random_split(all_ds, [split, rem])
    feat_size = ds[0][0].shape[-1]
    print(feat_size)

    if params.nn_flag:
        model = nn.Sequential(
            nn.Flatten(), nn.Linear(feat_size, 256), nn.Linear(128, 10)
        )

        trainer = Trainer(
            None,
            0,
            ds,
            test_ds,
            10,
            params.batch_size,
            params.nclients,
            params.fraction,
            params.rounds,
            params.epochs,
            params.gpu,
            params.verbose,
            params.nn_flag,
            model,
            params.expt,
        )
    else:
        click.echo("Creating HD embedding model")
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
            params.nn_flag,
            expt=params.expt,
        )
    click.echo("Running federated learning on the PAMAP Dataset")
    trainer.train()


@main.command()
@click.option(
    "-r",
    "--root",
    type=click.Path(exists=True, dir_okay=True),
    default="/home/the-noetic/cookiejar/data",
    help="Root directory of the data",
)
@pass_params
def mnist(params, root):

    click.echo("Loading MNIST dataset")
    transforms = tf.Compose([tf.ToTensor()])
    ds = MNIST(root, train=True, download=True, transform=transforms)
    test_ds = MNIST(root, train=False, download=True, transform=transforms)

    if params.nn_flag:
        model = nn.Sequential(nn.Flatten(), nn.Linear(784, 512), nn.Linear(512, 10))

        trainer = Trainer(
            None,
            0,
            ds,
            test_ds,
            10,
            params.batch_size,
            params.nclients,
            params.fraction,
            params.rounds,
            params.epochs,
            params.gpu,
            params.verbose,
            params.nn_flag,
            model,
            params.expt,
        )
        click.echo("running fl on MNIST NN")
        trainer.train()

    else:
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
            10,
            params.batch_size,
            params.nclients,
            params.fraction,
            params.rounds,
            params.epochs,
            params.gpu,
            params.verbose,
            params.nn_flag,
            expt=params.expt,
        )
        click.echo("Running federated learning on the MNIST Dataset")
        trainer.train()


@main.command()
@click.option(
    "-r",
    "--root",
    type=click.Path(exists=True, dir_okay=True),
    default="/home/the-noetic/cookiejar/data",
    help="Root directory of the data",
)
@pass_params
def face(params, root):
    click.echo("Loading FACE dataset")
    transforms = tf.Compose([tf.ToTensor()])
    all_ds = FaceDataset(root)
    split = int(0.75 * len(all_ds))
    rem = len(all_ds) - split
    ds, test_ds = random_split(all_ds, [split, rem])
    # ds = FaceDataset(root, train=True, transform=None)
    # test_ds = FaceDataset(root, train=False, transform=None)
    feat_size = 608

    if params.nn_flag:
        model = nn.Sequential(nn.Flatten(), nn.Linear(feat_size, 512), nn.Linear(512, 2))

        trainer = Trainer(
            None,
            0,
            ds,
            test_ds,
            10,
            params.batch_size,
            params.nclients,
            params.fraction,
            params.rounds,
            params.epochs,
            params.gpu,
            params.verbose,
            params.nn_flag,
            model,
            params.expt,
        )
        click.echo("running fl on MNIST NN")
        trainer.train()

    else:
        click.echo("Creating HD embedding model")
        embedding = nn.Sequential(
            nn.Flatten(), hd.embeddings.Projection(feat_size, params.dim)
        )

        trainer = Trainer(
            embedding,
            params.dim,
            ds,
            test_ds,
            10,
            params.batch_size,
            params.nclients,
            params.fraction,
            params.rounds,
            params.epochs,
            params.gpu,
            params.verbose,
            params.nn_flag,
            expt=params.expt,
        )
        click.echo("Running federated learning on the FACE Dataset")
        trainer.train()
