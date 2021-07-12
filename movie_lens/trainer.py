import ast
import logging
import typing
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from movie_lens import config, utils


@dataclass
class MovieGenreTrainer:
    dataset_path: Path
    df: typing.Optional[pd.DataFrame] = None
    feature_cols = ["title", "overview"]
    target_col: str = "genres"
    test_size: float = 0.2
    random_state: int = 47
    max_features: int = 1000

    def load_data(self):
        return pd.read_csv(self.dataset_path)

    def train(self):
        if not self.dataset_path.exists():  # Download dataset if it doesn't exist
            utils.download_dataset(config.DATASET_URL, config.DATA_DIR)

        if self.df is None:
            self.df = self.load_data()

        df = self.preprocess_data(self.df)

        # convert the generes into multi-label format
        labeler, labels = self.extract_genres(df)

        # combine title and overview columns
        features = df[self.feature_cols].apply(" ".join, axis=1)
        # build and train the model
        model, metrics = self.train_model(features, labels, labeler.classes_)
        # save model
        self.save_model(model, labeler)
        return metrics

    def preprocess_data(self, df):
        print(df.head())
        df.info()
        # From the given problem statement, we need only the following columns
        # title, overview and genre
        selected_cols = self.feature_cols + [self.target_col]
        df = df[selected_cols]
        # check missing values
        print("\nMissing values:\n", df.isna().sum())
        # drop the missing values since its text data and not appropriate to
        # make imputations for the missing values.
        df = df.dropna(subset=self.feature_cols)
        # let check some values in the genres column
        print(df[self.target_col])
        # The genres is a list of dictionaries, so will convert into a list of genres names
        df[self.target_col] = (
            df[self.target_col]
            .apply(ast.literal_eval)
            .apply(
                lambda genre: [x["name"] for x in genre]
                if isinstance(genre, list)
                else []
            )
        )
        # select the first genre as the genre for the movie
        df[self.target_col] = df[self.target_col].apply(
            lambda x: x[0] if len(x) > 0 else None
        )
        print(df[self.target_col])
        # We have some rows with no genres, will drop those to have a clean data
        df = df.dropna(subset=[self.target_col])
        # check missing values
        print("\nMissing values:\n", df.isna().sum())
        return df

    def extract_genres(self, df):
        labeler = LabelEncoder()
        labels = labeler.fit_transform(df[self.target_col])

        return labeler, labels

    def train_model(self, X, y, column_names):
        """
        Build and train RandomForestClassifier to predict movie genre
        Args:
            X ():
            y ():
            column_names ():

        Returns:

        """
        # split the dataset into training and test set
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        print("Training data size:", X_train.shape, y_train.shape)
        print("Test data size:", X_test.shape, y_test.shape)
        # We'll convert the title and overview features into TF-IDF
        # features using sklearn's TfidfVectorizer module
        tf_idf = TfidfVectorizer(
            max_features=self.max_features, stop_words="english", lowercase=True
        )

        # define RandomForestClassifier
        rf = RandomForestClassifier(random_state=self.random_state)

        pipe = Pipeline([("text_transform", tf_idf), ("clf", rf)])
        print(pipe)

        logging.info("Training RandomForestClassifier...")
        pipe.fit(X_train, y_train)

        metrics = self.model_reports(
            pipe, X_train, y_train, X_test, y_test, column_names
        )
        return pipe, metrics

    @staticmethod
    def model_reports(model, X_train, y_train, X_test, y_test, target_names):
        print("Classification Report")

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        print(
            "Training:\n",
            classification_report(
                y_true=y_train, y_pred=train_pred, target_names=target_names
            ),
        )
        print(
            "Validation:\n",
            classification_report(
                y_true=y_test, y_pred=test_pred, target_names=target_names
            ),
        )

        print("---Accuracy---")

        train_acc = accuracy_score(y_true=y_train, y_pred=train_pred)
        test_acc = accuracy_score(y_true=y_test, y_pred=test_pred)
        print("Traning: {:.2f}%".format(train_acc * 100))
        print("Validation: {:.2f}%".format(test_acc * 100))
        return train_acc, test_acc

    @staticmethod
    def save_model(model, labeler):
        logging.info(f"Saving trained model to {config.MOVIE_GENRE_CLF_PATH}")
        joblib.dump(
            {"model": model, "labeler": labeler},
            config.MOVIE_GENRE_CLF_PATH,
            compress=True,
        )
