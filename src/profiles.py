"""Loading the fictional household profiles shipped in data/."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.budget import Household

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROFILES_CSV = DATA_DIR / "example_households.csv"


def load_profiles(path: Path = PROFILES_CSV) -> dict[str, Household]:
    """Return {profile name: Household}. Every row is validated."""
    df = pd.read_csv(path)
    return {row["name"]: Household.from_mapping(row.to_dict()) for _, row in df.iterrows()}


def profile_descriptions(path: Path = PROFILES_CSV) -> dict[str, str]:
    df = pd.read_csv(path)
    return dict(zip(df["name"], df["description"]))
