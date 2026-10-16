"""
Automated downloader utility for LICHEN pretrained model weights.
Fetches model weights from Zenodo with integrity verification.
"""

import os
import urllib.request
import hashlib
import zipfile

ZENODO_WEIGHTS_URL = "https://zenodo.org/api/records/15917096/files/Model.zip/content"
EXPECTED_MD5 = "44b58244a4563cbc7b7b0fee1d0480a9"

def ensure_model_weights(model_path=None, target_dir="model", verbose=True) -> str:
    """
    Ensures that the model weights exist locally. If not found, downloads
    and extracts them from Zenodo with MD5 checksum verification.
    
    Parameters
    ----------
    model_path : str, optional
        User-specified path to model_weights.pt. If None or non-existent,
        falls back to default target_dir/model_weights.pt.
    target_dir : str, default 'model'
        Directory where model_weights.pt should be placed.
    verbose : bool, default True
        Whether to print progress messages.

    Returns
    -------
    str : Absolute or verified relative path to model_weights.pt.
    """
    # 1. If explicit path is provided and exists, use it
    if model_path and os.path.exists(model_path):
        return model_path

    # 2. Check default target location
    default_path = os.path.join(target_dir, "model_weights.pt")
    if os.path.exists(default_path):
        return default_path

    # Check parent directory or package relative if run from a subfolder (e.g. notebooks)
    alt_parent_path = os.path.join("..", target_dir, "model_weights.pt")
    if os.path.exists(alt_parent_path):
        return alt_parent_path

    # 3. Not found: initiate automatic download
    os.makedirs(target_dir, exist_ok=True)
    zip_dest = os.path.join(target_dir, "Model.zip")

    if verbose:
        print(f"Model weights not detected locally. Downloading from Zenodo (~94 MB)...")
        print(f"Source: {ZENODO_WEIGHTS_URL}")

    try:
        def _reporthook(block_num, block_size, total_size):
            if verbose and total_size > 0:
                downloaded = block_num * block_size
                percent = min(100.0, downloaded * 100.0 / total_size)
                if block_num % 1000 == 0 or percent >= 100:
                    print(f"\rDownloading weights: {percent:.1f}% ({downloaded // (1024*1024)} MB / {total_size // (1024*1024)} MB)", end="")

        urllib.request.urlretrieve(ZENODO_WEIGHTS_URL, zip_dest, reporthook=_reporthook)
        if verbose:
            print("\nDownload complete. Verifying MD5 checksum...")

        # 4. Verify MD5 Checksum
        hasher = hashlib.md5()
        with open(zip_dest, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()

        if digest != EXPECTED_MD5:
            raise ValueError(
                f"Checksum mismatch for downloaded weights!\n"
                f"Expected: {EXPECTED_MD5}\n"
                f"Got:      {digest}\n"
                f"The downloaded archive may be corrupted."
            )

        if verbose:
            print("Checksum verified. Extracting weights archive...")

        # 5. Extract weights
        with zipfile.ZipFile(zip_dest, "r") as zip_ref:
            zip_ref.extractall(target_dir)

        # Handle nested folder if archive contains 'Model/model_weights.pt'
        nested_file = os.path.join(target_dir, "Model", "model_weights.pt")
        if os.path.exists(nested_file):
            if os.path.exists(default_path):
                os.remove(default_path)
            os.replace(nested_file, default_path)
            try:
                os.rmdir(os.path.join(target_dir, "Model"))
            except OSError:
                pass

        # Cleanup zip file
        if os.path.exists(zip_dest):
            os.remove(zip_dest)

        if verbose:
            print(f"Model weights successfully installed at: {default_path}")

        return default_path

    except Exception as e:
        if os.path.exists(zip_dest):
            os.remove(zip_dest)
        raise RuntimeError(
            f"Failed to automatically download and verify model weights: {e}\n"
            f"Please download manually from https://doi.org/10.5281/zenodo.15917096 "
            f"and place model_weights.pt into '{default_path}'."
        ) from e
