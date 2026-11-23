"""Metric suite and the statistics used for the falsifier.

Positive class is SAFE (+1). A false SAFE is an UNSAFE load case predicted SAFE, which is the
dangerous design error, so its rate is reported separately.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score)


def classification_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    unsafe = y_true == -1
    false_safe = np.sum(np.logical_and(unsafe, y_pred == 1))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "error": 1 - accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, pos_label = 1, zero_division = 0),
        "precision": precision_score(y_true, y_pred, pos_label = 1, zero_division = 0),
        "recall": recall_score(y_true, y_pred, pos_label = 1, zero_division = 0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "false_safe_rate": false_safe / max(int(np.sum(unsafe)), 1),
    }


def test_set_metrics(y_true, y_pred, band_mask):
    """Metrics on the random test part (prefix test_) and the boundary band (prefix band_).

    band_error is the main metric of the study.
    """
    band_mask = np.asarray(band_mask, dtype = bool)
    random_mask = np.logical_not(band_mask)
    results = {}
    for name, value in classification_metrics(np.asarray(y_true)[random_mask], np.asarray(y_pred)[random_mask]).items():
        results[f"test_{name}"] = value
    for name, value in classification_metrics(np.asarray(y_true)[band_mask], np.asarray(y_pred)[band_mask]).items():
        results[f"band_{name}"] = value
    return results


def mean_confidence_interval(values, confidence = 0.95):
    values = np.asarray(values, dtype = float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float(np.mean(values)) if len(values) else np.nan, np.nan
    half_width = stats.t.ppf(0.5 + confidence / 2, len(values) - 1) * stats.sem(values)
    return float(np.mean(values)), float(half_width)


def summarise(results, metric, group_columns = ("model", "train_size")):
    """Mean, standard deviation and 95 percent CI half width of a metric over seeds."""
    rows = []
    for keys, group in results.groupby(list(group_columns)):
        mean, half_width = mean_confidence_interval(group[metric])
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_columns, keys))
        row.update({"mean": mean, "std": float(group[metric].std(ddof = 1)), "ci95": half_width, "runs": len(group)})
        rows.append(row)
    return pd.DataFrame(rows)


def paired_comparison(results, model_a, model_b, metric = "band_error", pair_columns = ("train_size", "train_seed")):
    """Paired differences (model_a minus model_b) on identical training sets.

    Returns one row per training size and one pooled row, with the mean difference, its 95
    percent CI, a paired t-test p value and a Wilcoxon signed rank p value.
    """
    first = results[results["model"] == model_a].set_index(list(pair_columns))[metric]
    second = results[results["model"] == model_b].set_index(list(pair_columns))[metric]
    joined = pd.concat([first.rename("a"), second.rename("b")], axis = 1).dropna()
    joined["difference"] = joined["a"] - joined["b"]

    rows = []
    groups = list(joined.groupby(level = 0)) + [("pooled", joined)]
    for size, group in groups:
        differences = group["difference"].to_numpy()
        mean, half_width = mean_confidence_interval(differences)
        if len(differences) > 1 and np.any(differences != 0):
            t_p = stats.ttest_rel(group["a"], group["b"]).pvalue
            try:
                wilcoxon_p = stats.wilcoxon(differences).pvalue
            except ValueError:
                wilcoxon_p = np.nan
        else:
            t_p = np.nan
            wilcoxon_p = np.nan
        rows.append({
            "train_size": size,
            "pairs": len(differences),
            f"mean_{metric}_{model_a}": float(group["a"].mean()),
            f"mean_{metric}_{model_b}": float(group["b"].mean()),
            "mean_difference": mean,
            "ci95_low": mean - half_width if np.isfinite(half_width) else np.nan,
            "ci95_high": mean + half_width if np.isfinite(half_width) else np.nan,
            "wins_for_a": int(np.sum(differences < 0)),
            "ties": int(np.sum(differences == 0)),
            "t_test_p": t_p,
            "wilcoxon_p": wilcoxon_p,
        })
    return pd.DataFrame(rows)


def kernel_diagnostics(gram, labels = None):
    """Mean off-diagonal value, smallest eigenvalue and (optionally) kernel target alignment."""
    gram = np.asarray(gram, dtype = float)
    off_diagonal = gram[np.logical_not(np.eye(len(gram), dtype = bool))]
    symmetric = (gram + gram.T) / 2
    result = {
        "mean_offdiag": float(off_diagonal.mean()),
        "std_offdiag": float(off_diagonal.std()),
        "min_eigenvalue": float(np.linalg.eigvalsh(symmetric).min()),
    }
    if labels is not None:
        target = np.outer(labels, labels)
        result["target_alignment"] = float(np.sum(symmetric * target) / (np.linalg.norm(symmetric) * np.linalg.norm(target)))
    return result


def mean_ci_table(results, metric_names, group_columns = ("model", "train_size")):
    """Wide table of 'mean +/- ci95' strings, one row per group."""
    pieces = []
    for metric_name in metric_names:
        summary = summarise(results, metric_name, group_columns)
        summary[metric_name] = [f"{mean:.3f} +/- {ci:.3f}" for mean, ci in zip(summary["mean"], summary["ci95"])]
        pieces.append(summary.set_index(list(group_columns))[[metric_name]])
    return pd.concat(pieces, axis = 1)
