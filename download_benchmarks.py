import io
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlopen

ZIP_URL = "https://github.com/LLM4OR/LLM4OR/archive/refs/heads/master.zip"
WANTED_PREFIX = "LLM4OR-master/static/datasets/"
OUT_DIR = Path("benchmarks")


def download_all_datasets():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with urlopen(ZIP_URL) as resp:
        zip_data = resp.read()

    with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
        matched = False
        for member in zf.namelist():
            if member.startswith(WANTED_PREFIX) and not member.endswith("/"):
                matched = True
                rel_path = member[len(WANTED_PREFIX):]
                out_file = OUT_DIR / rel_path
                out_file.parent.mkdir(parents=True, exist_ok=True)

                with zf.open(member) as src, open(out_file, "wb") as dst:
                    dst.write(src.read())

    if not matched:
        raise RuntimeError("Did not find datasets in the downloaded archive.")

    print(f"Done. Dataset saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    download_all_datasets()