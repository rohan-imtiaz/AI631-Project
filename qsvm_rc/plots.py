"""Plots shared by several notebooks."""

import matplotlib.pyplot as plt
import numpy as np

from . import config
from . import metrics
from . import rc_capacity


def learning_curve_plot(results, metric, models = None, title = None, axis = None, labels = None):
    """Mean over seeds with a 95 percent CI band, training size on a log2 axis."""
    axis = plt.gca() if axis is None else axis
    models = list(results["model"].unique()) if models is None else models
    summary = metrics.summarise(results[results["model"].isin(models)], metric)
    for model in models:
        part = summary[summary["model"] == model].sort_values("train_size")
        label = labels.get(model, model) if labels else model
        axis.plot(part["train_size"], part["mean"], marker = "o", label = label)
        axis.fill_between(part["train_size"], part["mean"] - part["ci95"], part["mean"] + part["ci95"], alpha = 0.15)
    axis.set_xscale("log", base = 2)
    axis.set_xticks(sorted(results["train_size"].unique()))
    axis.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axis.set_xlabel("training size")
    axis.set_ylabel(metric)
    axis.grid(alpha = 0.3)
    if title:
        axis.set_title(title)
    axis.legend(fontsize = 8)
    return axis


def eccentricity_grid_for_plot(section_key, resolution = 60):
    section = config.SECTIONS[section_key]
    ex_values = np.linspace(0, section["width_x_mm"] / 2, resolution)
    ey_values = np.linspace(0, section["depth_y_mm"] / 2, resolution)
    ex_grid, ey_grid = np.meshgrid(ex_values, ey_values)
    return ex_grid, ey_grid


def true_safe_mask(section_key, pu_kn, ex_grid, ey_grid):
    capacity = rc_capacity.biaxial_capacity(config.SECTIONS[section_key], ex_grid.ravel(), ey_grid.ravel())
    return (pu_kn <= capacity["phi_pni_kN"]).reshape(ex_grid.shape)


def safe_region_figure(section_key, scaler, predictors, pu_levels, resolution = 60):
    """Predicted SAFE region in the ex-ey plane for fixed Pu, one column per model.

    predictors: dict name -> callable(angles) returning +1 / -1. The black line is the true boundary.
    """
    ex_grid, ey_grid = eccentricity_grid_for_plot(section_key, resolution)
    names = list(predictors)
    figure, axes = plt.subplots(len(pu_levels), len(names), figsize = (3.2 * len(names), 2.8 * len(pu_levels)),
                                squeeze = False, sharex = True, sharey = True)
    for row, pu_kn in enumerate(pu_levels):
        raw = np.column_stack([np.full(ex_grid.size, pu_kn), ex_grid.ravel(), ey_grid.ravel()])
        angles = scaler.transform(raw)
        truth = true_safe_mask(section_key, pu_kn, ex_grid, ey_grid)
        for column, name in enumerate(names):
            axis = axes[row, column]
            predicted = (np.asarray(predictors[name](angles)) == 1).reshape(ex_grid.shape)
            agreement = np.mean(predicted == truth)
            axis.contourf(ex_grid, ey_grid, predicted.astype(float), levels = [-0.5, 0.5, 1.5],
                          colors = ["#f4c7c3", "#c6e5c3"])
            if truth.any() and not truth.all():
                axis.contour(ex_grid, ey_grid, truth.astype(float), levels = [0.5], colors = "k", linewidths = 1.2)
            axis.set_title(f"{name}\nPu = {pu_kn:.0f} kN, agree {agreement:.3f}", fontsize = 8)
            if row == len(pu_levels) - 1:
                axis.set_xlabel("ex (mm)")
            if column == 0:
                axis.set_ylabel("ey (mm)")
    figure.suptitle("Green: predicted SAFE, red: predicted UNSAFE, black line: true boundary", fontsize = 9)
    figure.tight_layout()
    return figure
