import click

from movie_lens import config
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
    if not title:  # check if title is empty
        raise ValueError(
            "Title must provided. Run `python main.py movie_classifier --help` for more details"
        )

    if not description:  # check if description is empty
        raise ValueError(
            "Description must provided. Run `python main.py movie_classifier --help` for more details"
        )

    click.echo(
        f"Predicting movie genre with <title: {title}> "
        f"and <description: {description}>"
    )
    clf = MovieGenreClassifier(model_path=config.MOVIE_GENRE_CLF_PATH)
    output = clf.predict(title, description)
    click.echo(output)


@cli.command()
def train():
    """
    Train movie genre classification model
    Returns:

    """
    click.echo("Start tarining model")
    MovieGenreClassifier.train(config.DATASET_PATH)
    click.echo("Done training model")


if __name__ == "__main__":
    cli()
