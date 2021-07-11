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
from sklearn.preprocessing import MultiLabelBinarizer

from movie_lens import config, utils


@dataclass
class MovieGenreTrainer:
    dataset_path: Path
    df: typing.Optional[pd.DataFrame] = None
    test_size: float = 0.2
    random_state: int = 47

    def train(self):
        if not self.dataset_path.exists():  # Download dataset if it doesn't exist
            utils.download_dataset(config.DATASET_URL, config.DATA_DIR)

        self.df = pd.read_csv(self.dataset_path)

        df = self.preprocess_data(self.df)

        # convert the generes into multi-label format
        labeler, labels = self.extract_genres(df)

        # combine title and overview columns
        features = df["title"] + df["overview"]
        # build and train the model
        model = self.train_model(features, labels, labeler.classes_)
        # save model
        logging.info(f"Saving trained model to {config.MOVIE_GENRE_CLF_PATH}")
        joblib.dump(
            {"pipeline": model, "labeler": labeler}, config.MOVIE_GENRE_CLF_PATH
        )

    @staticmethod
    def preprocess_data(df):
        print(df.head())
        df.info()
        # From the given problem statement, we need only the following columns
        # title, overview and genre
        df = df[["title", "overview", "genres"]]
        # check missing values
        print("\nMissing values:\n", df.isna().sum())
        # drop the missing values since its text data and not appropriate to
        # make imputations for the missing values.
        df = df.dropna(subset=["title", "overview"])
        # let check some values in the genres column
        print(df["genres"])
        # The genres is a list of dictionaries, so will convert into a list of genres names
        df["genres"] = (
            df["genres"]
            .apply(ast.literal_eval)
            .apply(
                lambda genre: [x["name"] for x in genre]
                if isinstance(genre, list)
                else []
            )
        )
        df["genres"] = df["genres"].apply(lambda x: x if len(x) > 0 else None)
        print(df["genres"])
        # We have some rows with no genres, will drop those to have a clean data
        df = df.dropna(subset=["genres"])
        # check missing values
        print("\nMissing values:\n", df.isna().sum())
        return df

    @staticmethod
    def extract_genres(df):
        labeler = MultiLabelBinarizer()
        labels = labeler.fit_transform(df["genres"])

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
            max_features=1000, stop_words="english", lowercase=True
        )

        # define RandomForestClassifier
        rf = RandomForestClassifier(random_state=self.random_state)

        pipe = Pipeline([("text_transform", tf_idf), ("clf", rf)])
        print(pipe)

        logging.info("Training RandomForestClassifier...")
        pipe.fit(X_train, y_train)

        self.model_reports(pipe, X_train, y_train, X_test, y_test, column_names)
        return pipe

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

        print("Accuracy")

        train_acc = accuracy_score(y_true=y_train, y_pred=train_pred)
        test_acc = accuracy_score(y_true=y_test, y_pred=test_pred)
        print("Traning: ", train_acc)
        print("Validation: ", test_acc)
        return train_acc, test_acc
