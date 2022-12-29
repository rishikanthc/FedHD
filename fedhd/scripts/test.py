import click
import torchhd as hd

from fedhd.client import Client
from fedhd.fl_trainer import Trainer


class Config(object):
    def __init__(self):
        self.epochs = None
        self.batch_size = None
        self.dataset = None
        self.gpu = None
        self.dim = None


pass_config = click.make_pass_decorator(Config, ensure=True)


@click.group()
@click.option("-e", "--epochs", default=2, help="# of epochs to train client")
@click.option("-bs", "--batch_size", default=5, help="batch size for client")
@click.option("-ds", "--dataset", default="mnist", help="Choose dataset")
@click.option("-g", "--gpu", is_flag=True, help="Run on GPU")
@click.option("-D", "--dim", default=2000, help="Hypervector dimensionality")
@click.option(
    "-v", "--verbose", is_flag=True, help="Verbose mode enables debug messages"
)
@pass_config
def cli(config, epochs, batch_size, dataset, gpu, dim, verbose):
    config.epochs = epochs
    config.batch_size = batch_size
    config.dataset = dataset
    config.gpu = gpu
    config.dim = dim
    config.verbose = verbose


@cli.command()
@pass_config
def client_test(config):
    click.echo("Testing client training")

    ds = hd.datasets.ISOLET(
        "/home/paperspace/codezone/data/mnist", train=True, download=True
    )
    embedding = hd.embeddings.Projection(617, config.dim)
    cv = hd.empty_hv(26, config.dim)
    client = Client(
        embedding,
        cv,
        ds,
        26,
        config.epochs,
        config.batch_size,
        config.gpu,
        config.verbose,
    )
    client.train()


@cli.command()
@click.option("-nc", "--nclients", default=10, help="# of clients")
@click.option(
    "-f",
    "--fraction",
    default=0.2,
    help="Fraction of clients involved in training rounds",
)
@click.option("-r", "--rounds", default=100, help="# of rounds of communication")
@pass_config
def data_test(config, nclients, fraction, rounds):
    click.echo("Testing data splitting")

    ds = hd.datasets.ISOLET(
        "/home/paperspace/codezone/data/mnist", train=True, download=True
    )
    embedding = hd.embeddings.Projection(617, config.dim)

    trainer = Trainer(
        embedding,
        config.dim,
        ds,
        None,
        26,
        config.batch_size,
        nclients,
        fraction,
        rounds,
        config.epochs,
        config.gpu,
        config.verbose,
    )
    del trainer


@cli.command()
@click.option("-nc", "--nclients", default=10, help="# of clients")
@click.option(
    "-f",
    "--fraction",
    default=0.2,
    help="Fraction of clients involved in training rounds",
)
@click.option("-r", "--rounds", default=100, help="# of rounds of communication")
@pass_config
def fl_test(config, nclients, fraction, rounds):
    click.echo("Testing FL pipeline")

    ds = hd.datasets.ISOLET(
        "/home/paperspace/codezone/data/mnist", train=True, download=True
    )
    test_ds = hd.datasets.ISOLET(
        "/home/paperspace/codezone/data/mnist", train=False, download=True
    )
    embedding = hd.embeddings.Projection(617, config.dim)

    trainer = Trainer(
        embedding,
        config.dim,
        ds,
        test_ds,
        26,
        config.batch_size,
        nclients,
        fraction,
        rounds,
        config.epochs,
        config.gpu,
        config.verbose,
    )
    trainer.train()
