import io
import logging
import zipfile

import requests


def download_dataset(url, dest_dir):
    logging.info(f"Downloading dataset into {dest_dir}")
    response = requests.get(url, allow_redirects=True, stream=True)

    zip_ref = zipfile.ZipFile(io.BytesIO(response.content))
    zip_ref.extractall(dest_dir)
