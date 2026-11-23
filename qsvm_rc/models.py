"""Classical baselines, the precomputed-kernel QSVM, and the learning curve driver.

All models see the same scaled inputs (angles in [0, pi]), the same training rows and the same
inner stratified folds, and they are tuned on balanced accuracy over those inner folds only.
"""

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from . import config
from . import metrics
from . import quantum_kernels

warnings.filterwarnings("ignore", category = ConvergenceWarning)


def inner_folds(y_train, seed):
    """Stratified inner folds. The same seed and labels give the same folds for every model."""
    smallest_class = int(np.min(np.bincount((np.asarray(y_train) + 1) // 2)))
    splits = int(np.clip(smallest_class, 2, config.INNER_FOLDS))
    return StratifiedKFold(n_splits = splits, shuffle = True, random_state = seed)


def classical_model_and_grid(name, seed, class_weight = None):
    """Estimator plus tuning grid for each classical baseline."""
    if name == "majority":
        return DummyClassifier(strategy = "most_frequent"), None
    if name == "logistic":
        return (LogisticRegression(max_iter = 5000, class_weight = class_weight),
                {"C": [0.01, 0.1, 1, 10, 100]})
    if name == "rbf_svm":
        return (SVC(kernel = "rbf", class_weight = class_weight),
                {"C": config.SVM_C_GRID, "gamma": config.RBF_GAMMA_GRID})
    if name == "mlp":
        # Two hidden layers, sigmoid output, log loss (binary cross entropy). lbfgs suits small data.
        return (MLPClassifier(solver = "lbfgs", max_iter = 2000, alpha = 1e-3, random_state = seed),
                {"hidden_layer_sizes": [(16, 16), (32, 32)], "activation": ["relu", "tanh"]})
    if name == "mlp_small":
        # One hidden layer of 3 units: 3 * 3 + 3 + 3 + 1 = 16 parameters, close to the 14 of the optional VQC
        return (MLPClassifier(hidden_layer_sizes = (3,), activation = "tanh", solver = "lbfgs",
                              max_iter = 2000, random_state = seed),
                {"alpha": [1e-4, 1e-2]})
    raise ValueError(f"Unknown classical model {name}")


def fit_classical(name, x_train, y_train, seed, class_weight = None):
    estimator, grid = classical_model_and_grid(name, seed, class_weight)
    if grid is None:
        return estimator.fit(x_train, y_train), {}, np.nan
    search = GridSearchCV(estimator, grid, scoring = "balanced_accuracy", cv = inner_folds(y_train, seed),
                          n_jobs = 1, error_score = 0)
    search.fit(x_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_


def classical_fit_predict(name, class_weight = None):
    """Callable (x_train, y_train, x_test, seed) -> (predictions, info) for the learning curve driver."""
    def fit_predict(x_train, y_train, x_test, seed):
        model, best_params, cv_score = fit_classical(name, x_train, y_train, seed, class_weight)
        info = {"cv_balanced_accuracy": cv_score, "best_params": str(best_params)}
        return model.predict(x_test), info
    return fit_predict


def matched_rbf_fit_predict(bandwidth, class_weight = None):
    """RBF SVM with gamma = c^2 / 4, the width at which exp(-gamma |d|^2) matches prod cos^2(c d_j / 2) for small d.

    Only C is tuned (5 settings), because the width is set by the angle kernel.
    """
    gamma = bandwidth ** 2 / 4

    def fit_predict(x_train, y_train, x_test, seed):
        search = GridSearchCV(SVC(kernel = "rbf", gamma = gamma, class_weight = class_weight),
                              {"C": config.SVM_C_GRID}, scoring = "balanced_accuracy",
                              cv = inner_folds(y_train, seed), n_jobs = 1, error_score = 0)
        search.fit(x_train, y_train)
        return search.predict(x_test), {"cv_balanced_accuracy": search.best_score_,
                                        "best_params": str({"gamma": gamma, **search.best_params_})}
    return fit_predict


def encode(angles, spec, bandwidth):
    """Select and order the features named in spec["features"], then multiply by the bandwidth."""
    features = list(spec.get("features", [0, 1, 2]))
    return bandwidth * np.asarray(angles, dtype = float)[:, features]


class ExactStateCache:
    """Caches exact states of a fixed set of angles (the test set) for each spec and bandwidth."""

    def __init__(self, angles):
        self.angles = np.asarray(angles, dtype = float)
        self.cache = {}

    def states(self, spec, bandwidth):
        key = (quantum_kernels.spec_name(spec), tuple(spec.get("features", [0, 1, 2])), float(bandwidth))
        if key not in self.cache:
            self.cache[key] = quantum_kernels.exact_states(encode(self.angles, spec, bandwidth), spec)
        return self.cache[key]


def tune_qsvm(spec, x_train, y_train, seed, bandwidth_grid = None, c_grid = None, class_weight = None):
    """Inner CV over (bandwidth, C) on exact kernels. Returns the best setting and the CV table."""
    bandwidth_grid = config.BANDWIDTH_GRID if bandwidth_grid is None else bandwidth_grid
    c_grid = config.SVM_C_GRID if c_grid is None else c_grid
    folds = list(inner_folds(y_train, seed).split(x_train, y_train))
    rows = []
    for bandwidth in bandwidth_grid:
        states = quantum_kernels.exact_states(encode(x_train, spec, bandwidth), spec)
        gram = np.abs(np.conj(states) @ states.T) ** 2
        for c_value in c_grid:
            scores = []
            for fit_rows, check_rows in folds:
                classifier = SVC(kernel = "precomputed", C = c_value, class_weight = class_weight)
                classifier.fit(gram[np.ix_(fit_rows, fit_rows)], y_train[fit_rows])
                predicted = classifier.predict(gram[np.ix_(check_rows, fit_rows)])
                scores.append(balanced_accuracy_score(y_train[check_rows], predicted))
            rows.append({"bandwidth": bandwidth, "C": c_value, "cv_balanced_accuracy": float(np.mean(scores))})
    table = pd.DataFrame(rows)
    best = table.loc[table["cv_balanced_accuracy"].idxmax()]
    return float(best["bandwidth"]), float(best["C"]), float(best["cv_balanced_accuracy"]), table


def fit_exact_qsvm(spec, x_train, y_train, bandwidth, c_value, class_weight = None):
    """Fit an SVC on the exact training kernel. Returns the classifier and the training states."""
    states_train = quantum_kernels.exact_states(encode(x_train, spec, bandwidth), spec)
    gram = np.abs(np.conj(states_train) @ states_train.T) ** 2
    classifier = SVC(kernel = "precomputed", C = c_value, class_weight = class_weight).fit(gram, y_train)
    return classifier, states_train, gram


def qsvm_fit_predict(spec, test_cache, bandwidth_grid = None, c_grid = None, class_weight = None,
                     fixed_bandwidth = None, fixed_c = None):
    """Callable for the learning curve driver. Tunes (bandwidth, C) unless fixed values are given."""
    def fit_predict(x_train, y_train, x_test, seed):
        if fixed_bandwidth is not None and fixed_c is not None:
            bandwidth, c_value, cv_score = fixed_bandwidth, fixed_c, np.nan
        else:
            band_grid = [fixed_bandwidth] if fixed_bandwidth is not None else bandwidth_grid
            reg_grid = [fixed_c] if fixed_c is not None else c_grid
            bandwidth, c_value, cv_score, _ = tune_qsvm(spec, x_train, y_train, seed, band_grid, reg_grid, class_weight)
        classifier, states_train, gram = fit_exact_qsvm(spec, x_train, y_train, bandwidth, c_value, class_weight)
        states_test = test_cache.states(spec, bandwidth)
        test_gram = np.abs(np.conj(states_test) @ states_train.T) ** 2
        info = {"cv_balanced_accuracy": cv_score, "bandwidth": bandwidth, "C": c_value,
                "best_params": str({"bandwidth": bandwidth, "C": c_value})}
        info.update(metrics.kernel_diagnostics(gram, y_train))
        return classifier.predict(test_gram), info
    return fit_predict


def run_learning_curve(model_functions, frame, splits, scaler, sizes = None, seeds = None, strategy = "random",
                       save_path = None, extra_columns = None, verbose = True):
    """Train every model on every saved training set and score it on the fixed test set.

    model_functions: dict name -> fit_predict(x_train, y_train, x_test, seed).
    Rows are appended to save_path after every run, so a crash keeps what was finished.
    """
    from . import data_utils

    sizes = config.TRAIN_SIZES if sizes is None else sizes
    seeds = config.TRAIN_SEEDS if seeds is None else seeds
    test_rows, band_mask = data_utils.test_indices(splits)
    x_test, y_test = data_utils.features_and_labels(frame, test_rows, scaler)

    rows = []
    for size in sizes:
        for train_seed in seeds:
            train_rows = data_utils.training_indices(splits, size, train_seed, strategy)
            x_train, y_train = data_utils.features_and_labels(frame, train_rows, scaler)
            for name, fit_predict in model_functions.items():
                start = time.time()
                predictions, info = fit_predict(x_train, y_train, x_test, config.run_seed(config.SEED, train_seed))
                row = {"section": splits["section_key"], "model": name, "strategy": strategy,
                       "train_size": size, "train_seed": train_seed, "seconds": time.time() - start}
                row.update(extra_columns or {})
                row.update(metrics.test_set_metrics(y_test, predictions, band_mask))
                row.update(info)
                rows.append(row)
                if verbose:
                    band_error = row["band_error"]
                    balanced_accuracy = row["test_balanced_accuracy"]
                    false_safe = row["test_false_safe_rate"]
                    seconds = row["seconds"]
                    print(f"{name:<24} n={size:<4} seed={train_seed}  band_error={band_error:.3f}  "
                          f"test_bal_acc={balanced_accuracy:.3f}  false_safe={false_safe:.3f}  ({seconds:.1f} s)")
                if save_path is not None:
                    pd.DataFrame(rows).to_csv(save_path, index = False)
    return pd.DataFrame(rows)


def exact_qsvm_predictor(spec, x_train, y_train, bandwidth, c_value, class_weight = None, return_decision = False):
    """Fit once, then return a callable(angles) giving labels (or decision values) for any points."""
    classifier, states_train, _ = fit_exact_qsvm(spec, x_train, y_train, bandwidth, c_value, class_weight)

    def predict(angles):
        states = quantum_kernels.exact_states(encode(angles, spec, bandwidth), spec)
        gram = np.abs(np.conj(states) @ states_train.T) ** 2
        return classifier.decision_function(gram) if return_decision else classifier.predict(gram)
    return predict


def most_common_setting(results, model_name, train_size, columns = ("bandwidth", "C")):
    """Most frequent tuned setting over seeds at one training size (ties go to the lower band error)."""
    part = results[(results["model"] == model_name) & (results["train_size"] == train_size)]
    grouped = part.groupby(list(columns)).agg(count = ("band_error", "size"), band_error = ("band_error", "mean")).reset_index()
    grouped = grouped.sort_values(["count", "band_error"], ascending = [False, True])
    return {column: float(grouped.iloc[0][column]) for column in columns}
