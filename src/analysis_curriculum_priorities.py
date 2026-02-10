"""Analyze curriculum priority rankings from a Qualtrics export."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .load_qualtrics import load_qualtrics_csv

CSV_PATH = "data/raw/Alternative CPA Pathways Survey_December 31, 2025_09.45.csv"
REPORT_PATH = "reports/curriculum_priority_differences.md"
FIGURES_DIR = Path("reports/figures")

CONFIG = {
    "segmentation_columns": {
        "undergrad_vs_grad": None,
        "online_vs_inperson": None,
        "cpa_intent": None,
        "awareness": None,
    }
}

RANKING_COLUMN_PATTERN = re.compile(r"^(Q\d+)_\d+$")

DISCIPLINE_KEYWORDS = [
    "auditing",
    "tax",
    "financial",
    "managerial",
    "information systems",
    "data analytics",
    "other business",
]


@dataclass
class SegmentSpec:
    label: str
    series: pd.Series
    group_order: List[str]


def normalize_label(text: str, fallback: str) -> str:
    if not text:
        return fallback
    cleaned = text.strip()
    for sep in (":", "-", "–", "—"):
        if sep in cleaned:
            cleaned = cleaned.split(sep)[-1].strip()
    return cleaned or fallback


def find_column_by_question_text(
    question_map: Dict[str, str], keyword_sets: Iterable[Iterable[str]]
) -> Optional[str]:
    best_col = None
    best_score = 0
    for col, text in question_map.items():
        lower_text = text.lower()
        for keywords in keyword_sets:
            if all(keyword in lower_text for keyword in keywords):
                score = sum(lower_text.count(keyword) for keyword in keywords)
                if score > best_score:
                    best_score = score
                    best_col = col
    return best_col


def detect_ranking_block(
    columns: Iterable[str], question_map: Dict[str, str]
) -> Tuple[List[str], Dict[str, str]]:
    grouped: Dict[str, List[str]] = {}
    for col in columns:
        match = RANKING_COLUMN_PATTERN.match(col)
        if not match:
            continue
        prefix = match.group(1)
        grouped.setdefault(prefix, []).append(col)

    candidates: List[Tuple[str, List[str], bool, int]] = []
    for prefix, cols in grouped.items():
        combined_text = " ".join(question_map.get(col, "") for col in cols).lower()
        has_rank = "rank" in combined_text
        discipline_matches = sum(keyword in combined_text for keyword in DISCIPLINE_KEYWORDS)
        if has_rank and discipline_matches > 0:
            candidates.append((prefix, cols, "rank the following disciplines" in combined_text, discipline_matches))

    if not candidates:
        return [], {}

    candidates.sort(
        key=lambda item: (
            item[2],
            item[3],
            len(item[1]),
        ),
        reverse=True,
    )

    chosen_cols = sorted(candidates[0][1])
    label_map = {
        col: normalize_label(question_map.get(col, ""), col)
        for col in chosen_cols
    }
    return chosen_cols, label_map


def to_numeric_rank(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def compute_metrics(df: pd.DataFrame, rank_cols: List[str], label_map: Dict[str, str]) -> pd.DataFrame:
    rows = []
    for col in rank_cols:
        values = to_numeric_rank(df[col])
        non_missing = values.dropna()
        n = int(non_missing.shape[0])
        mean_rank = float(non_missing.mean()) if n else np.nan
        median_rank = float(non_missing.median()) if n else np.nan
        pct_rank1 = float((non_missing == 1).mean() * 100) if n else np.nan
        pct_top3 = float((non_missing <= 3).mean() * 100) if n else np.nan
        rows.append(
            {
                "Discipline": label_map.get(col, col),
                "N": n,
                "MeanRank": mean_rank,
                "MedianRank": median_rank,
                "PctRank1": pct_rank1,
                "PctTop3": pct_top3,
            }
        )
    metrics = pd.DataFrame(rows)
    metrics = metrics.sort_values("MeanRank", ascending=True, na_position="last")
    return metrics


def format_metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.copy()
    for col in ["MeanRank", "MedianRank", "PctRank1", "PctTop3"]:
        formatted[col] = formatted[col].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    return formatted


def map_undergrad_grad(series: pd.Series) -> pd.Series:
    def mapper(value: object) -> Optional[str]:
        if pd.isna(value):
            return None
        text = str(value).lower()
        if "undergrad" in text:
            return "Undergraduate"
        if "graduate" in text or "grad" in text:
            return "Graduate"
        return None

    return series.map(mapper)


def map_online_inperson(series: pd.Series) -> pd.Series:
    def mapper(value: object) -> Optional[str]:
        if pd.isna(value):
            return None
        text = str(value).lower()
        if "online" in text:
            return "Online"
        if "in-person" in text or "in person" in text or "campus" in text:
            return "In-person"
        return None

    return series.map(mapper)


def map_awareness(series: pd.Series) -> pd.Series:
    def mapper(value: object) -> Optional[str]:
        if pd.isna(value):
            return None
        text = str(value).lower()
        if text in {"yes", "y", "aware"} or "yes" in text:
            return "Aware"
        if text in {"no", "n", "not aware"} or "no" in text:
            return "Not aware"
        return None

    return series.map(mapper)


def infer_likert_score(value: object) -> Optional[float]:
    if pd.isna(value):
        return None
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).lower()
    mapping = {
        "extremely likely": 7,
        "very likely": 6,
        "likely": 5,
        "somewhat likely": 4,
        "neither": 3,
        "neutral": 3,
        "somewhat unlikely": 2,
        "unlikely": 2,
        "very unlikely": 1,
        "extremely unlikely": 1,
    }
    for key, score in mapping.items():
        if key in text:
            return float(score)
    match = re.search(r"\d+(?:\.\d+)?", text)
    if match:
        return float(match.group())
    return None


def map_cpa_intent(series: pd.Series) -> pd.Series:
    scores = series.map(infer_likert_score)
    numeric_values = scores.dropna().unique().tolist()
    if not numeric_values:
        return pd.Series([None] * len(series), index=series.index)

    levels = sorted(set(numeric_values))
    if len(levels) >= 5:
        high_levels = set(levels[-2:])
        low_levels = set(levels[:2])
    else:
        high_levels = {levels[-1]}
        low_levels = {levels[0]}

    def mapper(score: Optional[float]) -> Optional[str]:
        if score is None or pd.isna(score):
            return None
        if score in high_levels:
            return "High"
        if score in low_levels:
            return "Low"
        return "Mid/Neutral"

    return scores.map(mapper)


def add_stat_tests(
    df: pd.DataFrame,
    series: pd.Series,
    rank_col: str,
    group_order: List[str],
) -> Tuple[Optional[float], Optional[float]]:
    import importlib.util

    if importlib.util.find_spec(\"scipy\") is None:
        return None, None

    from scipy import stats

    values = to_numeric_rank(df[rank_col])
    grouped = [
        values[series == group].dropna().values
        for group in group_order
        if group in series.values
    ]
    if len(grouped) < 2:
        return None, None
    if len(grouped) == 2:
        u_stat, p_value = stats.mannwhitneyu(grouped[0], grouped[1], alternative="two-sided")
        n1, n2 = len(grouped[0]), len(grouped[1])
        if n1 * n2 == 0:
            return p_value, None
        rbc = 1 - (2 * u_stat) / (n1 * n2)
        return p_value, rbc

    h_stat, p_value = stats.kruskal(*grouped)
    n_total = sum(len(group) for group in grouped)
    k = len(grouped)
    if n_total <= k:
        return p_value, None
    eta_sq = max((h_stat - k + 1) / (n_total - k), 0)
    return p_value, eta_sq


def fdr_bh(p_values: List[Optional[float]]) -> List[Optional[float]]:
    clean = [(i, p) for i, p in enumerate(p_values) if p is not None]
    if not clean:
        return [None] * len(p_values)
    order = sorted(clean, key=lambda x: x[1])
    m = len(order)
    adjusted = [None] * len(p_values)
    prev = 0
    for rank, (idx, p) in enumerate(order, start=1):
        adj = p * m / rank
        adj = min(adj, 1.0)
        adj = max(adj, prev)
        adjusted[idx] = adj
        prev = adj
    return adjusted


def build_segment_table(
    df: pd.DataFrame,
    rank_cols: List[str],
    label_map: Dict[str, str],
    spec: SegmentSpec,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    metrics_by_group = {}
    for group in spec.group_order:
        group_df = df[spec.series == group]
        if group_df.empty:
            continue
        metrics = compute_metrics(group_df, rank_cols, label_map)
        metrics_by_group[group] = metrics.set_index("Discipline")

    if not metrics_by_group:
        return pd.DataFrame(), None

    combined = pd.concat(metrics_by_group, axis=1)

    if len(metrics_by_group) == 2:
        group_a, group_b = [g for g in spec.group_order if g in metrics_by_group][:2]
        diff = (
            metrics_by_group[group_a]["MeanRank"]
            - metrics_by_group[group_b]["MeanRank"]
        )
        combined[("Difference", "MeanRank") ] = diff

    combined = combined.swaplevel(axis=1).sort_index(axis=1, level=0)
    combined = combined.reset_index()

    stats_rows = []
    p_values = []
    for col in rank_cols:
        p_val, effect = add_stat_tests(df, spec.series, col, spec.group_order)
        p_values.append(p_val)
        stats_rows.append(
            {
                "Discipline": label_map.get(col, col),
                "p_value": p_val,
                "effect_size": effect,
            }
        )

    stats_df = pd.DataFrame(stats_rows)
    stats_df["p_adj"] = fdr_bh(p_values)

    return combined, stats_df


def plot_overall_mean_rank(metrics: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(metrics["Discipline"], metrics["MeanRank"], color="#4C72B0")
    ax.invert_yaxis()
    ax.set_xlabel("Mean Rank (lower = higher priority)")
    ax.set_title("Overall Mean Rank by Discipline")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_segment_mean_rank(
    combined: pd.DataFrame,
    spec: SegmentSpec,
    output_path: Path,
) -> None:
    mean_cols = [
        col for col in combined.columns if isinstance(col, tuple) and col[1] == "MeanRank"
    ]
    if not mean_cols:
        return
    disciplines = combined["Discipline"]
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.8 / len(mean_cols)
    y_positions = np.arange(len(disciplines))
    for idx, col in enumerate(mean_cols):
        group = col[0]
        offsets = y_positions + idx * bar_width
        ax.barh(offsets, combined[col], height=bar_width, label=group)
    ax.set_yticks(y_positions + bar_width * (len(mean_cols) - 1) / 2)
    ax.set_yticklabels(disciplines)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Rank")
    ax.set_title(f"Mean Rank by {spec.label}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_report(
    overall_metrics: pd.DataFrame,
    segment_outputs: List[Tuple[SegmentSpec, pd.DataFrame, Optional[pd.DataFrame]]],
) -> str:
    lines = []
    lines.append("# Curriculum Priority Differences\n")
    lines.append(
        "Ranks are interpreted as 1 = highest priority. Metrics are computed from non-missing rankings.\n"
    )
    lines.append("## Overall priorities\n")
    lines.append(format_metrics_table(overall_metrics).to_markdown(index=False))
    lines.append("\n![](figures/overall_mean_rank.png)\n")

    for spec, combined, stats_df in segment_outputs:
        lines.append(f"## {spec.label}\n")
        if combined.empty:
            lines.append("No data available for this segment.\n")
            continue
        formatted = combined.copy()
        for col in formatted.columns:
            if isinstance(col, tuple) and col[1] in {"MeanRank", "MedianRank", "PctRank1", "PctTop3"}:
                formatted[col] = formatted[col].map(
                    lambda x: "" if pd.isna(x) else f"{x:.2f}"
                )
        if ("Difference", "MeanRank") in formatted.columns:
            formatted[("Difference", "MeanRank")] = formatted[("Difference", "MeanRank")].map(
                lambda x: "" if pd.isna(x) else f"{x:.2f}"
            )
        formatted.columns = [
            " ".join([str(part) for part in col if part]) if isinstance(col, tuple) else col
            for col in formatted.columns
        ]
        lines.append(formatted.to_markdown(index=False))
        chart_name = f"figures/{spec.label.lower().replace(' ', '_')}_mean_rank.png"
        lines.append(f"\n![]({chart_name})\n")
        if stats_df is not None and stats_df["p_value"].notna().any():
            stats_display = stats_df.copy()
            stats_display["p_value"] = stats_display["p_value"].map(
                lambda x: "" if pd.isna(x) else f"{x:.4f}"
            )
            stats_display["p_adj"] = stats_display["p_adj"].map(
                lambda x: "" if pd.isna(x) else f"{x:.4f}"
            )
            stats_display["effect_size"] = stats_display["effect_size"].map(
                lambda x: "" if pd.isna(x) else f"{x:.3f}"
            )
            lines.append("\nStatistical tests (nonparametric with FDR correction):\n")
            lines.append(stats_display.to_markdown(index=False))
    return "\n".join(lines)


def main() -> None:
    plt.switch_backend("Agg")
    df, question_map = load_qualtrics_csv(CSV_PATH)

    rank_cols, label_map = detect_ranking_block(df.columns, question_map)
    if not rank_cols:
        print("Warning: Could not detect ranking block columns.")
        return

    overall_metrics = compute_metrics(df, rank_cols, label_map)

    segmentation_columns = CONFIG["segmentation_columns"]
    segments: List[SegmentSpec] = []
    warnings = []

    undergrad_col = segmentation_columns.get("undergrad_vs_grad") or find_column_by_question_text(
        question_map,
        [
            ["undergraduate", "graduate"],
            ["current status"],
        ],
    )
    if undergrad_col:
        series = map_undergrad_grad(df[undergrad_col])
        segments.append(SegmentSpec("Undergrad vs Graduate", series, ["Undergraduate", "Graduate"]))
    else:
        warnings.append("Undergrad vs Graduate")

    online_col = segmentation_columns.get("online_vs_inperson") or find_column_by_question_text(
        question_map,
        [
            ["online", "in-person"],
            ["online", "in person"],
        ],
    )
    if online_col:
        series = map_online_inperson(df[online_col])
        segments.append(SegmentSpec("Online vs In-person", series, ["Online", "In-person"]))
    else:
        warnings.append("Online vs In-person")

    intent_col = segmentation_columns.get("cpa_intent") or find_column_by_question_text(
        question_map,
        [["likely", "cpa"], ["pursue", "cpa"]],
    )
    if intent_col:
        series = map_cpa_intent(df[intent_col])
        segments.append(SegmentSpec("CPA Intent Level", series, ["High", "Mid/Neutral", "Low"]))
    else:
        warnings.append("CPA Intent Level")

    awareness_col = segmentation_columns.get("awareness") or find_column_by_question_text(
        question_map,
        [["aware", "alternative pathway"], ["aware", "pathway"]],
    )
    if awareness_col:
        series = map_awareness(df[awareness_col])
        segments.append(SegmentSpec("Awareness of Alternative Pathway", series, ["Aware", "Not aware"]))
    else:
        warnings.append("Awareness of Alternative Pathway")

    if warnings:
        print(
            "Warning: Could not find segmentation columns for: "
            + ", ".join(warnings)
            + ". Configure CONFIG['segmentation_columns'] to override."
        )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_overall_mean_rank(overall_metrics, FIGURES_DIR / "overall_mean_rank.png")

    segment_outputs = []
    for spec in segments:
        combined, stats_df = build_segment_table(df, rank_cols, label_map, spec)
        segment_outputs.append((spec, combined, stats_df))
        if not combined.empty:
            chart_path = FIGURES_DIR / f"{spec.label.lower().replace(' ', '_')}_mean_rank.png"
            plot_segment_mean_rank(combined, spec, chart_path)

    report = build_report(overall_metrics, segment_outputs)
    Path(REPORT_PATH).write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
