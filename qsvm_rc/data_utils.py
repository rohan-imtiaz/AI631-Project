"""Loading, labels, the capacity ratio, fixed splits and the leakage-free angle scaler."""

import json

import numpy as np
import pandas as pd

from . import config
from . import rc_capacity


def _find_column(columns, starts, contains = None):
    """First column whose lower-case name starts with one of `starts` (and contains `contains`)."""
    for column in columns:
        lowered = column.lower().strip()
        if any(lowered.startswith(prefix) for prefix in starts):
            if contains is None or contains in lowered:
                return column
    return None


def standardise_columns(raw):
    """Rename the dataset columns to pu_kN, ex_mm, ey_mm, status, whatever their exact spelling."""
    columns = list(raw.columns)
    mapping = {
        _find_column(columns, ["pu"]): "pu_kN",
        _find_column(columns, ["ex", "e_x"]): "ex_mm",
        _find_column(columns, ["ey", "e_y"]): "ey_mm",
        _find_column(columns, ["status", "label", "class", "safe"]): "status",
    }
    if None in mapping:
        raise KeyError(f"Could not recognise the columns {columns}. Rename them to pu_kN, ex_mm, ey_mm, status.")
    frame = raw.rename(columns = mapping)
    frame["status"] = frame["status"].astype(str).str.upper().str.strip()
    return frame


def _phi_pni_from_extended(extended_path, frame):
    """phi Pni from the extended file if a matching column exists, otherwise None."""
    if not extended_path.exists():
        return None, "extended file not found"
    extended = pd.read_csv(extended_path)
    lowered = {column: column.lower().replace(" ", "").replace("_", "") for column in extended.columns}
    phi_columns = [column for column, name in lowered.items() if "pni" in name and "phi" in name]
    pni_columns = [column for column, name in lowered.items() if "pni" in name and "phi" not in name and "1/" not in name]
    if len(extended) != len(frame):
        return None, "extended file has a different number of rows"
    if phi_columns:
        return extended[phi_columns[0]].to_numpy(dtype = float), f"extended column '{phi_columns[0]}'"
    if pni_columns:
        return config.PHI * extended[pni_columns[0]].to_numpy(dtype = float), f"{config.PHI} x extended column '{pni_columns[0]}'"
    return None, "no Pni column recognised in the extended file"


def load_section(section_key, capacity_source = "auto"):
    """Dataset for one section with label (+1 SAFE, -1 UNSAFE), phi Pni and the capacity ratio r = Pu / (phi Pni).

    capacity_source: "auto" uses the extended file when a phi Pni (or Pni) column is found and
    falls back to the project capacity code, "code" always uses the project capacity code.
    """
    section = config.SECTIONS[section_key]
    frame = standardise_columns(pd.read_csv(section["dataset_path"]))
    frame["label"] = np.where(frame["status"] == "SAFE", 1, -1)

    phi_pni = None
    source = "project capacity code"
    if capacity_source == "auto":
        phi_pni, message = _phi_pni_from_extended(section["extended_path"], frame)
        if phi_pni is not None:
            source = message
        else:
            print(f"Capacity: {message}, so phi Pni is recomputed with the project capacity code.")
    if phi_pni is None:
        phi_pni = rc_capacity.biaxial_capacity(section, frame["ex_mm"].to_numpy(), frame["ey_mm"].to_numpy())["phi_pni_kN"]

    frame["phi_pni_kN"] = phi_pni
    frame["ratio"] = frame["pu_kN"] / frame["phi_pni_kN"]
    frame.attrs["capacity_source"] = source
    frame.attrs["section_key"] = section_key
    return frame


def label_agreement(frame):
    """Share of rows whose stored label agrees with r <= 1 from the capacity values used here."""
    implied = np.where(frame["ratio"] <= 1 + 1e-9, 1, -1)
    return float(np.mean(implied == frame["label"].to_numpy()))


class DomainAngleScaler:
    """Maps each feature linearly from its design domain onto [0, pi].

    The bounds are the limits of the load and eccentricity grids (known before any data is
    drawn), so nothing is fitted on test data and small training sets cannot squeeze the range.
    """

    def __init__(self, lower_bounds, upper_bounds):
        self.lower_bounds = np.asarray(lower_bounds, dtype = float)
        self.upper_bounds = np.asarray(upper_bounds, dtype = float)

    def transform(self, features):
        features = np.asarray(features, dtype = float)
        unit = (features - self.lower_bounds) / (self.upper_bounds - self.lower_bounds)
        return np.pi * np.clip(unit, 0, 1)

    def to_dict(self):
        return {"lower_bounds": self.lower_bounds.tolist(), "upper_bounds": self.upper_bounds.tolist()}


def domain_scaler(section_key, frame = None):
    """Scaler from the grid limits: Pu from the dataset grid, ex and ey from 0 to the half dimensions."""
    section = config.SECTIONS[section_key]
    if frame is not None:
        pu_low = float(frame["pu_kN"].min())
        pu_high = float(frame["pu_kN"].max())
    else:
        loads = rc_capacity.axial_load_grid(section)
        pu_low = float(loads.min())
        pu_high = float(loads.max())
    return DomainAngleScaler([pu_low, 0, 0], [pu_high, section["width_x_mm"] / 2, section["depth_y_mm"] / 2])


def _stratified_draw(rng, candidate_indices, labels, size, safe_fraction):
    """Draw `size` indices from the candidates, keeping the given SAFE fraction."""
    candidate_indices = np.asarray(candidate_indices)
    safe_candidates = candidate_indices[labels[candidate_indices] == 1]
    unsafe_candidates = candidate_indices[labels[candidate_indices] == -1]
    number_safe = int(np.clip(np.round(size * safe_fraction), 2, size - 2))
    chosen_safe = rng.choice(safe_candidates, size = number_safe, replace = False)
    chosen_unsafe = rng.choice(unsafe_candidates, size = size - number_safe, replace = False)
    return np.sort(np.concatenate([chosen_safe, chosen_unsafe]))


def make_fixed_splits(frame, section_key, test_random_size = config.TEST_RANDOM_SIZE,
                      test_band_size = config.TEST_BAND_SIZE, seed = config.SEED):
    """Fixed test set (random part plus boundary band part) and every training set of the protocol.

    Training sets never contain test points. For each strategy, size and seed the indices are
    drawn once and saved, so every model in every notebook trains on exactly the same rows.
    """
    rng = np.random.default_rng(config.run_seed(seed, 7))
    labels = frame["label"].to_numpy()
    distance = np.abs(frame["ratio"].to_numpy() - 1)
    all_indices = np.arange(len(frame))
    safe_fraction = float(np.mean(labels == 1))

    band_candidates = all_indices[distance <= config.TEST_BAND_HALF_WIDTH]
    test_band = np.sort(rng.choice(band_candidates, size = min(test_band_size, len(band_candidates)), replace = False))
    remaining = np.setdiff1d(all_indices, test_band)
    test_random = _stratified_draw(rng, remaining, labels, test_random_size, safe_fraction)
    test_all = np.union1d(test_band, test_random)
    pool = np.setdiff1d(all_indices, test_all)

    pool_band = pool[distance[pool] <= config.TRAIN_BAND_HALF_WIDTH]
    training_sets = {}
    for strategy in config.SAMPLING_STRATEGIES:
        for size in config.TRAIN_SIZES:
            for train_seed in config.TRAIN_SEEDS:
                draw_rng = np.random.default_rng(config.run_seed(seed, size, train_seed + 1, config.SAMPLING_STRATEGIES.index(strategy) + 1))
                if strategy == "random":
                    chosen = _stratified_draw(draw_rng, pool, labels, size, safe_fraction)
                elif strategy == "boundary":
                    chosen = _stratified_draw(draw_rng, pool_band, labels, size, safe_fraction)
                else:
                    half = size // 2
                    near = _stratified_draw(draw_rng, pool_band, labels, half, safe_fraction)
                    far_pool = np.setdiff1d(pool, near)
                    far = _stratified_draw(draw_rng, far_pool, labels, size - half, safe_fraction)
                    chosen = np.sort(np.concatenate([near, far]))
                training_sets[f"{strategy}|{size}|{train_seed}"] = chosen.tolist()

    splits = {
        "section_key": section_key,
        "seed": seed,
        "test_random": test_random.tolist(),
        "test_band": test_band.tolist(),
        "pool_size": int(len(pool)),
        "pool_band_size": int(len(pool_band)),
        "safe_fraction": safe_fraction,
        "test_band_half_width": config.TEST_BAND_HALF_WIDTH,
        "train_band_half_width": config.TRAIN_BAND_HALF_WIDTH,
        "training_sets": training_sets,
    }
    return splits


def save_splits(splits):
    path = config.section_results_dir(splits["section_key"], "splits") / "fixed_splits.json"
    with open(path, "w") as handle:
        json.dump(splits, handle)
    return path


def load_splits(section_key):
    path = config.section_results_dir(section_key, "splits") / "fixed_splits.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run 01_data/01_data_audit_and_fixed_splits.ipynb first "
                                "(or 01_data/02_extra_sections_generation.ipynb for the extra sections).")
    with open(path) as handle:
        return json.load(handle)


def training_indices(splits, size, train_seed, strategy = "random"):
    return np.asarray(splits["training_sets"][f"{strategy}|{size}|{train_seed}"])


def test_indices(splits):
    """All test rows, plus a boolean mask that marks the boundary band rows."""
    band = np.asarray(splits["test_band"])
    random_part = np.asarray(splits["test_random"])
    indices = np.concatenate([random_part, band])
    band_mask = np.concatenate([np.zeros(len(random_part), dtype = bool), np.ones(len(band), dtype = bool)])
    return indices, band_mask


def features_and_labels(frame, indices, scaler):
    """Scaled angles in [0, pi] and labels for the given rows."""
    rows = frame.iloc[np.asarray(indices)]
    return scaler.transform(rows[config.FEATURE_COLUMNS].to_numpy()), rows["label"].to_numpy()
