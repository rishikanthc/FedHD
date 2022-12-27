import click
import torchhd as hd

from fedhd.client import Client


@click.command()
@click.option("--epochs", default=2, help="# of epochs to train client")
@click.option("--batch_size", default=5, help="batch size for client")
@click.option("--dataset", default="mnist", help="Choose dataset")
@click.option(
    "--device", default="cpu", help="cuda for running on gpu and cpu otherwise"
)
@click.option("--dim", default=2000, help="Hypervector dimensionality")
def main(dim, dataset, epochs, batch_size, device):
    click.echo("successfully import Client")

    ds = hd.datasets.ISOLET(
        "/home/paperspace/codezone/data/mnist", train=True, download=True
    )
    embedding = hd.embeddings.Projection(617, dim)
    cv = hd.empty_hv(26, dim)

    client = Client(embedding, cv, ds, 26, epochs, batch_size, device)
    client.train()
