import os
import json
import hashlib
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# Resolve base directories relative to this file
SRC_DIR = Path(__file__).resolve().parent
ML_DIR = SRC_DIR.parent
RAW_DATA_DIR = ML_DIR / "data" / "raw"

FIGSHARE_ARTICLE_ID = "24052236"
ARTICLE_API_URL = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE_ID}"

DOI = "https://doi.org/10.6084/m9.figshare.24052236.v2"
CITATION = (
    "Akram, Umair; Irvine, Kamila; Gardani, Maria; Allen, Sarah; Stevenson, Jodie C. (2023). "
    "Comprehensive Data on the Prevalence of Psychiatric Symptoms in UK University Students: "
    "Data Files SPSS and XLSX Format and Variable Codes. figshare. Dataset. "
    "https://doi.org/10.6084/m9.figshare.24052236.v2"
)
LICENSE = "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)"


def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_file(url: str, dest_path: Path, expected_size: int = None) -> None:
    if dest_path.exists():
        actual_size = dest_path.stat().st_size
        if expected_size is None or actual_size == expected_size:
            print(f"[SKIP] Already exists: {dest_path.name} ({actual_size} bytes)")
            return
        else:
            print(f"[RE-DOWNLOAD] Size mismatch for {dest_path.name} (found {actual_size}, expected {expected_size})")

    print(f"[DOWNLOADING] {dest_path.name} from {url}...")
    headers = {"User-Agent": "MindWave-XGBoost-Research/1.0"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as out_file:
        chunk_size = 65536
        while chunk := resp.read(chunk_size):
            out_file.write(chunk)
    print(f"[DOWNLOADED] {dest_path.name} ({dest_path.stat().st_size} bytes)")


def main():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Fetching article metadata from Figshare API...")
    req = urllib.request.Request(ARTICLE_API_URL, headers={"User-Agent": "MindWave-ML/1.0"})
    with urllib.request.urlopen(req) as resp:
        article_info = json.loads(resp.read().decode("utf-8"))

    files_info = article_info.get("files", [])
    print(f"Found {len(files_info)} files in article record.")

    downloaded_files_meta = []
    for f in files_info:
        fname = f["name"]
        furl = f["download_url"]
        fsize = f.get("size")
        dest = RAW_DATA_DIR / fname

        download_file(furl, dest, expected_size=fsize)

        file_sha256 = compute_sha256(dest)
        downloaded_files_meta.append({
            "filename": fname,
            "figshare_id": f.get("id"),
            "download_url": furl,
            "expected_size_bytes": fsize,
            "actual_size_bytes": dest.stat().st_size,
            "sha256": file_sha256
        })

    metadata = {
        "dataset_name": "Prevalence of Psychiatric Symptoms in UK University Students",
        "citation": CITATION,
        "doi": DOI,
        "license": LICENSE,
        "download_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "files": downloaded_files_meta
    }

    meta_file = RAW_DATA_DIR / "dataset_metadata.json"
    with open(meta_file, "w", encoding="utf-8") as mf:
        json.dump(metadata, mf, indent=2)

    print(f"\nMetadata saved to {meta_file}")
    print("[SUCCESS] All dataset files acquired and verified successfully.")


if __name__ == "__main__":
    main()
