"""Download the public inputs into data/raw/ (about 4 MB zipped)."""
import io
import urllib.request
import zipfile
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
EHDP = "https://github.com/nycehs/All_EHDP_Data/raw/master/All_EHDP_data.zip"
UHF = "https://raw.githubusercontent.com/nycehs/NYC_geography/master/UHF42.geo.json"


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    print("Downloading NYC Environment & Health Data Portal export ...")
    with urllib.request.urlopen(EHDP) as r:
        zipfile.ZipFile(io.BytesIO(r.read())).extract("All_EHDP_data.csv", RAW)
    if not (RAW / "UHF42.geo.json").exists():
        print("Downloading UHF42 boundaries ...")
        urllib.request.urlretrieve(UHF, RAW / "UHF42.geo.json")
    print(f"Saved to {RAW}")


if __name__ == "__main__":
    main()
