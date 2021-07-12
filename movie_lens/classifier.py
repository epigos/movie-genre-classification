import typing
from dataclasses import dataclass
from pathlib import Path

import joblib

from movie_lens.trainer import MovieGenreTrainer


@dataclass
class MovieGenreClassifier:
    model_path: Path
    model: typing.Optional[typing.Any] = None
    labeler: typing.Optional[typing.Any] = None

    def __post_init__(self):
        self.load_model()

    def load_model(self):
        if not self.model_path.exists():
            raise RuntimeError("Model not found")

        model_dict = joblib.load(self.model_path)
        self.model = model_dict["model"]
        self.labeler = model_dict["labeler"]

    def predict(self, title: str, description: str) -> dict:
        """
        Predict movie genre using title and description of the movie
        Args:
            title (): the movie title
            description (): the movie description

        Returns:
        """
        input_text = f"{title} {description}"
        preds = self.model.predict([input_text])

        genres = self.labeler.inverse_transform(preds)
        return dict(title=title, description=description, genre=genres[0])

    @classmethod
    def train(cls, dataset_path):
        """
        Train movie genre classification model using the given dataset
        Args:
            dataset_path (): Path to dataset

        Returns:

        """
        trainer = MovieGenreTrainer(dataset_path=dataset_path)
        trainer.train()
