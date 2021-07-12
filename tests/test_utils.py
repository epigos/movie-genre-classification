from movie_lens import config, utils


def test_download_dataset():
    utils.download_dataset(config.DATASET_URL, config.DATA_DIR)
    assert config.DATASET_PATH.exists()
