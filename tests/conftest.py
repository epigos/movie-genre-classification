import pytest
from click.testing import CliRunner

from movie_lens import config
from movie_lens.classifier import MovieGenreClassifier


@pytest.fixture
def movie_classifier():
    clf = MovieGenreClassifier(model_path=config.MOVIE_GENRE_CLF_PATH)
    return clf


@pytest.fixture
def cli():
    runner = CliRunner()
    return runner
