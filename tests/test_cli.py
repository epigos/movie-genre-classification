from unittest.mock import patch

import pytest

from main import movie_classifier, train


def test_movie_classifier(cli, faker):
    title = faker.sentence()
    description = faker.paragraph()

    result = cli.invoke(
        movie_classifier,
        ["--title", title, "--description", description],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert title in result.output
    assert description in result.output
    assert "genre" in result.output


def test_invalid_input_movie_classifier(cli):
    # check no arguments
    result = cli.invoke(movie_classifier, [], catch_exceptions=False)

    assert result.exit_code == 2
    assert "Usage" in result.output  # assert usage information is printed to console

    # check empty arguments
    with pytest.raises(ValueError):
        cli.invoke(
            movie_classifier,
            ["--title", "", "--description", ""],
            catch_exceptions=False,
        )


def test_train_command(cli):
    with patch("movie_lens.trainer.MovieGenreTrainer.train") as mock_train:
        result = cli.invoke(train, [], catch_exceptions=False)

    assert result.exit_code == 0
    mock_train.assert_called()
    assert "Done training model" in result.output
