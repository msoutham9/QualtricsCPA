"""Utilities for loading Qualtrics CSV exports."""

from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd


def load_qualtrics_csv(path: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Load a Qualtrics CSV export that includes question text and ImportId rows.

    The CSV structure is assumed to be:
    - Row 0: column names
    - Row 1: question text
    - Row 2: ImportId metadata
    - Row 3+: responses

    Returns a tuple of (responses_df, question_map).
    """
    header_df = pd.read_csv(path, nrows=0)
    columns = [col.strip() for col in header_df.columns]

    meta_df = pd.read_csv(path, header=None, nrows=2)
    question_row = meta_df.iloc[0].tolist() if len(meta_df.index) > 0 else []

    question_map: Dict[str, str] = {}
    for idx, col in enumerate(columns):
        text = ""
        if idx < len(question_row) and pd.notna(question_row[idx]):
            text = str(question_row[idx]).strip()
        question_map[col] = text

    responses = pd.read_csv(path, skiprows=[1, 2])
    responses.columns = [col.strip() for col in responses.columns]

    drop_cols = [
        col
        for col in responses.columns
        if col.startswith("Unnamed") or responses[col].isna().all()
    ]
    if drop_cols:
        responses = responses.drop(columns=drop_cols)
        for col in drop_cols:
            question_map.pop(col, None)

    return responses, question_map
