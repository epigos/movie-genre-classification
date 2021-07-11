import click

from movie_lens.classifier import MovieGenreClassifier


@click.group()
def cli():
    pass


@cli.command("movie_classifier")
@click.option("--title", help="the movie title", required=True)
@click.option("--description", help="the movie description", required=True)
def movie_classifier(title, description):
    """
    Predict movie genre
    Returns:

    """
    click.echo(f"Predicting movie genre with <title: {title}> and <description: {description}>")
    return MovieGenreClassifier.predict(title, description)


@cli.command()
def train():
    """
    Train movie genre classification model
    Returns:

    """
    click.echo('Start tarining model')
    MovieGenreClassifier.train()


if __name__ == '__main__':
    cli()
