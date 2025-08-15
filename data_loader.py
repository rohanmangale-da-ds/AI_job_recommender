import json
import pandas as pd

def load_job_data(file_path: str) -> pd.DataFrame:
    """Load jobs from a JSON file and return as DataFrame."""
    with open(file_path, "r", encoding="utf-8") as f:
        jobs = json.load(f)
    return pd.DataFrame(jobs)
