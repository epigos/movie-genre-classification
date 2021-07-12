from pathlib import Path

import pytest


def test_valid_model_loading(movie_classifier):
    assert movie_classifier.model is not None
    assert movie_classifier.labeler is not None


def test_invalid_model_loading(movie_classifier):
    # load invalid model path
    movie_classifier.model_path = Path("/tmp/dummy.joblib")
    with pytest.raises(RuntimeError):
        movie_classifier.load_model()


def test_genre_predict(faker, movie_classifier):
    title = faker.sentence()
    description = faker.paragraph()

    output = movie_classifier.predict(title, description)
    assert output is not None
    assert output["title"] == title
    assert output["description"] == description
    assert output["genre"]
