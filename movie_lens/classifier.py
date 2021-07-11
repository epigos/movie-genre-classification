import joblib

from movie_lens import config


class MovieGenreClassifier:
    model = None

    @classmethod
    def predict(cls, title, description):
        if cls.model is None:
            cls.model = joblib.load(config.MOVIE_GENRE_CLF_PATH)

        preds = cls.model.predict([title + description])
        return preds

    @classmethod
    def train(cls):
        pass
