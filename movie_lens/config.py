import logging
from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "model"
MOVIE_GENRE_CLF_PATH = MODEL_DIR / "movie_genre.joblib"

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(asctime)s pid:%(process)s module:%(module)s %(message)s",
    datefmt="%d/%m/%y %H:%M:%S",
)
