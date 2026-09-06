"""
download_real_images.py
------------------------
Downloads REAL, freely-licensed breed photos from Wikipedia to replace the
placeholder example thumbnails, and (optionally) seeds a `raw_dataset/`
folder with one starter photo per breed.

WHY THIS IS A SEPARATE SCRIPT
This project's code was assembled in a sandboxed environment with no
internet access, so real photos couldn't be embedded directly into the
delivered zip. Run this script on YOUR machine (which has internet) and it
will fetch them for you.

WHERE THE IMAGES COME FROM
Each image is the lead photo of that breed's English Wikipedia article,
fetched via Wikipedia's public REST summary API
(https://en.wikipedia.org/api/rest_v1/page/summary/<title>). These images
are hosted on Wikimedia Commons and are almost always released under
Creative Commons (CC-BY-SA) or public-domain licenses that permit reuse
with attribution. Check each file's Commons page (linked in the script
output) for exact attribution requirements before using it outside a
personal/student project.

USAGE
    pip install requests
    python download_real_images.py                  # only the 6 sidebar examples
    python download_real_images.py --all             # all 30 breeds -> static/img/examples/
    python download_real_images.py --all --seed-dataset
        # also copies each downloaded photo into raw_dataset/<breed>/wikipedia_01.jpg
        # NOTE: one photo per breed is nowhere near enough to train a real
        # CNN - this just seeds the correct folder structure. You still need
        # to add many more images per breed yourself (see dataset/README.md).
"""

import argparse
import os
import re
import time

import requests

EXAMPLES_DIR = os.path.join("static", "img", "examples")
RAW_DATASET_DIR = "raw_dataset"

# breed_info.py key -> best-guess English Wikipedia article title
WIKI_TITLES = {
    # Cattle
    "Gir": "Gyr cattle",
    "Sahiwal": "Sahiwal cattle",
    "Red Sindhi": "Red Sindhi cattle",
    "Tharparkar": "Tharparkar cattle",
    "Rathi": "Rathi cattle",
    "Kankrej": "Kankrej",
    "Ongole": "Ongole cattle",
    "Hariana": "Hariana cattle",
    "Deoni": "Deoni cattle",
    "Khillari": "Khillari cattle",
    "Kangayam": "Kangayam cattle",
    "Vechur": "Vechur cattle",
    "Punganur": "Punganur cattle",
    "Amritmahal": "Amritmahal cattle",
    "Hallikar": "Hallikar",
    "Nagori": "Nagori cattle",
    "Malvi": "Malvi cattle",
    "Nimari": "Nimari cattle",
    "Dangi": "Dangi cattle",
    "Krishna Valley": "Krishna Valley cattle",
    # Buffalo
    "Murrah": "Murrah buffalo",
    "Jaffarabadi": "Jaffarabadi buffalo",
    "Mehsana": "Mehsana buffalo",
    "Surti": "Surti buffalo",
    "Nili-Ravi": "Nili-Ravi buffalo",
    "Bhadawari": "Bhadawari buffalo",
    "Nagpuri": "Nagpuri buffalo",
    "Toda": "Toda buffalo",
    "Pandharpuri": "Pandharpuri buffalo",
    "Banni": "Banni buffalo",
}

# The 6 breeds shown in the sidebar "Example Images" grid (must match the
# EXAMPLES array at the top of static/js/script.js).
SIDEBAR_EXAMPLES = ["Gir", "Ongole", "Murrah", "Jaffarabadi", "Kankrej", "Nili-Ravi"]

HEADERS = {"User-Agent": "BreedRecognitionStudentProject/1.0 (educational use)"}


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def fetch_wikipedia_image(title: str):
    """Return (image_bytes, source_url, commons_page_url) for a Wikipedia article's lead image."""
    api_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title)}"
    resp = requests.get(api_url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    image_info = data.get("originalimage") or data.get("thumbnail")
    if not image_info:
        raise ValueError(f"No image found on Wikipedia page '{title}'.")

    img_url = image_info["source"]
    img_resp = requests.get(img_url, headers=HEADERS, timeout=20)
    img_resp.raise_for_status()

    page_url = data.get("content_urls", {}).get("desktop", {}).get("page", api_url)
    return img_resp.content, img_url, page_url


def save_image(image_bytes: bytes, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(image_bytes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true",
                         help="Download all 30 breeds instead of just the 6 sidebar examples.")
    parser.add_argument("--seed-dataset", action="store_true",
                         help="Also copy each image into raw_dataset/<breed>/ to seed the training folder structure.")
    args = parser.parse_args()

    breeds = list(WIKI_TITLES.keys()) if args.all else SIDEBAR_EXAMPLES

    print(f"Downloading real Wikipedia photos for {len(breeds)} breed(s)...\n")
    ok, failed = [], []

    for breed in breeds:
        title = WIKI_TITLES[breed]
        try:
            img_bytes, img_url, page_url = fetch_wikipedia_image(title)

            # Save to the sidebar examples folder (used by the web UI)
            example_path = os.path.join(EXAMPLES_DIR, f"{slugify(breed)}.jpg")
            save_image(img_bytes, example_path)

            # Optionally seed the raw dataset folder structure for training
            if args.seed_dataset:
                seed_path = os.path.join(RAW_DATASET_DIR, breed, "wikipedia_01.jpg")
                save_image(img_bytes, seed_path)

            print(f"  [OK] {breed:16s} <- {page_url}")
            ok.append(breed)
            time.sleep(0.3)  # be polite to the API
        except Exception as e:
            print(f"  [FAIL] {breed:16s} - {e}")
            failed.append(breed)

    print(f"\nDone. {len(ok)} succeeded, {len(failed)} failed.")
    if failed:
        print("Failed breeds (placeholder graphics will remain for these):", ", ".join(failed))
    print(f"\nImages saved to: {os.path.abspath(EXAMPLES_DIR)}")
    if args.seed_dataset:
        print(f"Seed dataset saved to: {os.path.abspath(RAW_DATASET_DIR)}")
        print("Reminder: 1 image/breed is only a starter — add many more per breed "
              "before running train_model.py (see dataset/README.md).")

    print("\nLicensing: these photos come from Wikipedia/Wikimedia Commons and are "
          "typically CC-BY-SA or public domain. Check each article's image credit "
          "if you need formal attribution for your submission/report.")


if __name__ == "__main__":
    main()
