#!/usr/bin/env python3
import os
import json
import argparse
from kaggle.api.kaggle_api_extended import KaggleApi

'''
Download or upload Kaggle datasets or competitions.
This is for kaggle==1.15.6, due to issue with download competitions in latest vesrions.
Will have to only modify the kaggle.json path to ~/.config/kaggle/kaggle.json for newer versions.
'''


# ---------------------------------------------------------
# Set up Kaggle credentials
# ---------------------------------------------------------
def setup_kaggle_credentials():
    kaggle_dir = os.path.expanduser("~/.kaggle")
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")

    if not os.path.exists(kaggle_json):
        raise FileNotFoundError(
            "❌ kaggle.json not found in ~/.kaggle/. "
            "Place your kaggle.json there or copy it manually."
        )

    os.chmod(kaggle_json, 0o600)
    print("✓ Kaggle credentials validated.")


# ---------------------------------------------------------
# Download competition
# ---------------------------------------------------------
def download_competition(api, competition, out):
    os.makedirs(out, exist_ok=True)
    print(f"↓ Downloading competition '{competition}' → {out}")
    api.competition_download_files(competition, path=out)
    print("✓ Done")


# ---------------------------------------------------------
# Download dataset
# ---------------------------------------------------------
def download_dataset(api, dataset, out):
    os.makedirs(out, exist_ok=True)
    print(f"↓ Downloading dataset '{dataset}' → {out}")
    api.dataset_download_files(dataset, path=out, unzip=True)
    print("✓ Done")


# ---------------------------------------------------------
# Upload or update dataset
# ---------------------------------------------------------
def upload_dataset(api, folder, username, dataset_name, version_notes):
    os.makedirs(folder, exist_ok=True)
    metadata_path = os.path.join(folder, "dataset-metadata.json")

    # Prepare metadata
    metadata = {
        "title": dataset_name,
        "id": f"{username}/{dataset_name}",
        "licenses": [{"name": "CC0-1.0"}]
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Attempt creation first, fallback to updating
    try:
        print("↑ Creating new dataset…")
        api.dataset_create_new(folder, convert_to_csv=False, dir_mode="tar")
        print("✓ Dataset created")
    except Exception:
        print("↻ Dataset exists — uploading new version…")
        api.dataset_create_version(
            folder,
            version_notes=version_notes,
            convert_to_csv=False,
            dir_mode="tar",
        )
        print("✓ Dataset updated")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Kaggle dataset & competition helper"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ----------------- download competition -----------------
    dl_comp = subparsers.add_parser("download-competition")
    dl_comp.add_argument("--competition", required=True,
                         help="Competition slug (e.g. vesuvius-challenge-surface-detection)")
    dl_comp.add_argument("--out", default="data/",
                         help="Output folder")

    # ----------------- download dataset -----------------
    dl_ds = subparsers.add_parser("download-dataset")
    dl_ds.add_argument("--dataset", required=True,
                       help="Dataset slug (e.g. username/dataset-name)")
    dl_ds.add_argument("--out", default="data/",
                       help="Output folder")

    # ----------------- upload dataset -----------------
    up = subparsers.add_parser("upload-dataset")
    up.add_argument("--folder", required=True,
                    help="Folder that contains files to upload")
    up.add_argument("--username", required=True,
                    help="Your Kaggle username")
    up.add_argument("--dataset-name", required=True,
                    help="Dataset ID name (e.g. my-dataset)")
    up.add_argument("--notes", default="Update",
                    help="Version notes")

    args = parser.parse_args()

    setup_kaggle_credentials()

    api = KaggleApi()
    api.authenticate()

    if args.command == "download-competition":
        download_competition(api, args.competition, args.out)

    elif args.command == "download-dataset":
        download_dataset(api, args.dataset, args.out)

    elif args.command == "upload-dataset":
        upload_dataset(api, args.folder, args.username, args.dataset-name, args.notes)


if __name__ == "__main__":
    main()
