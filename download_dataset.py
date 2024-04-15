import io
import os
import sys
import zipfile

import requests
import requests_cache
from tqdm import tqdm


def main():
    zip_link = "http://www.mattmahoney.net/dc/enwik8.zip"
    data_folder = "dataset"
    cache_file = "download_cache"

    # Ensure the data folder exists
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    # Initialize requests_cache
    requests_cache.install_cache(os.path.join(data_folder, cache_file))

    # Download the ZIP file with progress bar
    response = requests.get(zip_link, stream=True)
    response.raise_for_status()

    # Get the total file size for the progress bar
    total_size = int(response.headers.get("content-length", 0))

    # Open the ZIP file from the content
    with open(os.path.join(data_folder, "enwik8.zip"), "wb") as file:
        with tqdm(
            total=total_size, unit="B", unit_scale=True, desc="Downloading"
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                file.write(data)
                pbar.update(len(data))

    # Open the cached file
    with open(os.path.join(data_folder, "enwik8.zip"), "rb") as file:
        # Open the ZIP file from the content
        with zipfile.ZipFile(io.BytesIO(file.read())) as zip_file:
            # Extract all contents to the data folder
            zip_file.extractall(data_folder)

    print("File downloaded and decompressed successfully.", file=sys.stderr)


if __name__ == "__main__":
    main()
