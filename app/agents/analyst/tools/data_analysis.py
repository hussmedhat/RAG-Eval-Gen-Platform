"""
Analyst Tool 4: Data Analysis

Performs structured statistical analysis over numeric data extracted
from documents (typically the output of the Table Extractor, or any
list of labeled numeric values pulled from evidence chunks). Handles
averages, percentages, rankings, and distribution stats — reusing the
Calculator's safe arithmetic rather than trusting the LLM to compute
values internally.
"""

import statistics
from dataclasses import dataclass, field

from app.agents.analyst.tools.calcaulator import average, percentage_change


@dataclass
class DataPoint:
    label: str          # e.g. a model name, a document source, a row identifier
    value: float


@dataclass
class DistributionStats:
    mean: float
    median: float
    stdev: float | None   # None if fewer than 2 points
    minimum: DataPoint
    maximum: DataPoint


@dataclass
class AnalysisResult:
    stats: DistributionStats
    ranking: list[DataPoint] = field(default_factory=list)  # sorted descending by value


def _to_datapoints(values: dict[str, float]) -> list[DataPoint]:
    if not values:
        raise ValueError("Cannot analyze an empty set of values.")
    return [DataPoint(label=k, value=v) for k, v in values.items()]


def compute_distribution(values: dict[str, float]) -> DistributionStats:
    """Computes mean/median/stdev/min/max over a labeled set of numbers."""
    points = _to_datapoints(values)
    nums = [p.value for p in points]

    mean_val = average(nums)
    median_val = statistics.median(nums)
    stdev_val = statistics.stdev(nums) if len(nums) > 1 else None

    minimum = min(points, key=lambda p: p.value)
    maximum = max(points, key=lambda p: p.value)

    return DistributionStats(
        mean=mean_val,
        median=median_val,
        stdev=stdev_val,
        minimum=minimum,
        maximum=maximum,
    )


def rank_values(values: dict[str, float], descending: bool = True) -> list[DataPoint]:
    """Ranks labeled values best-to-worst (or ascending if descending=False).
    Use descending=False for metrics where lower is better (e.g. latency, error rate)."""
    points = _to_datapoints(values)
    return sorted(points, key=lambda p: p.value, reverse=descending)


def analyze(values: dict[str, float], descending: bool = True) -> AnalysisResult:
    """Runs full distribution + ranking analysis over a labeled numeric set.
    e.g. analyze({"ModelA": 0.91, "ModelB": 0.87, "ModelC": 0.94})"""
    return AnalysisResult(
        stats=compute_distribution(values),
        ranking=rank_values(values, descending=descending),
    )


def compare_pair_percentage(baseline_label: str, baseline_value: float,
                             comparison_label: str, comparison_value: float) -> str:
    """Returns a human-readable percentage-change summary between two values,
    e.g. for 'how much better is Model B than Model A'."""
    change = percentage_change(baseline_value, comparison_value)
    direction = "higher" if change > 0 else "lower" if change < 0 else "unchanged"
    return f"{comparison_label} is {abs(change):.2f}% {direction} than {baseline_label}"