"""Download the StereoSet validation parquet files (public dev set) into data/raw/."""
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

for part in ("intersentence", "intrasentence"):
    cached = hf_hub_download(
        repo_id="McGill-NLP/stereoset",
        repo_type="dataset",
        filename=f"{part}/validation-00000-of-00001.parquet",
    )
    dest = RAW / f"stereoset_{part}_validation.parquet"
    shutil.copy(cached, dest)
    print(dest.name, dest.stat().st_size, "bytes")
