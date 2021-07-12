import tempfile
from pathlib import Path

import joblib

from movie_lens import config
from movie_lens.trainer import MovieGenreTrainer


def test_preprocess():
    """
    After preprocessing, there should be no missing values in df
    Returns:

    """
    trainer = MovieGenreTrainer(dataset_path=config.DATASET_PATH)

    df = trainer.load_data()
    df = trainer.preprocess_data(df)
    # check no missing data
    missing = df.isna().sum()
    assert all([v == 0 for v in missing.values])


def test_train(monkeypatch):
    trainer = MovieGenreTrainer(dataset_path=config.DATASET_PATH)

    df = trainer.load_data()
    trainer.df = df.sample(10000)

    with tempfile.NamedTemporaryFile(suffix=".joblib") as fp:
        model_path = Path(fp.name)
        monkeypatch.setattr(config, "MOVIE_GENRE_CLF_PATH", model_path)

        metrics = trainer.train()
        model_dict = joblib.load(model_path)

    assert metrics
    assert "model" in model_dict
    assert "labeler" in model_dict
