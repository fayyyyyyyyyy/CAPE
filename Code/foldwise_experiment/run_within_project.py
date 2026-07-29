import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from representation_foldwise import generate_fold_features


DEFAULT_PROJECTS = [
    "ant_main",
    "argouml",
    "gwtportlets",
    "javaclient",
    "jedit",
    "jgap",
    "jhotdraw",
    "jmeter_core",
    "JPMC",
    "log4j",
    "Mars",
    "Maze",
    "neuroph",
    "PDFBox",
    "tomcat",
    "wro4j",
    "Xerces",
    "xuml",
]

RF_PARAMETER_GRID = {
    "n_estimators": list(range(10, 71, 10)),
    "max_depth": list(range(3, 14, 2)),
    "min_samples_split": list(range(10, 201, 20)),
    "min_samples_leaf": list(range(10, 60, 10)),
}

METRIC_COLUMNS = [
    "precision",
    "recall",
    "f1",
    "auc",
    "accuracy",
    "mcc",
    "brier_score",
]


def parse_args():
    code_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run repeated CAPE within-project evaluation."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=code_root / "dataset_keyDesign_Sora",
        help="Directory containing one subdirectory per project.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=code_root / "foldwise_results",
        help="Directory for metrics and split manifests.",
    )
    parser.add_argument("--projects", nargs="+", help="Project names to evaluate.")
    parser.add_argument(
        "--projects-file",
        type=Path,
        help="Optional text file containing one project name per line.",
    )
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--variant", choices=("cape", "cape-a", "cape-n"), default="cape"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--n-jobs", type=int, default=-1)
    search_group = parser.add_mutually_exclusive_group()
    search_group.add_argument(
        "--grid-search",
        action="store_true",
        help="Enable Random Forest grid search.",
    )
    search_group.add_argument(
        "--no-grid-search",
        dest="grid_search",
        action="store_false",
        help="Disable Random Forest grid search.",
    )
    parser.set_defaults(grid_search=False)
    return parser.parse_args()


def resolve_projects(args):
    if args.projects_file is not None:
        projects = [
            line.strip()
            for line in args.projects_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    elif args.projects:
        projects = args.projects
    else:
        projects = DEFAULT_PROJECTS
    if not projects:
        raise ValueError("No projects were selected")
    return projects


def load_project_labels(project, dataset_root):
    process_path = Path(dataset_root) / project / "Process-Binary.csv"
    frame = pd.read_csv(process_path)
    if "KeyDesign" not in frame.columns:
        raise KeyError("KeyDesign column is missing from {}".format(process_path))
    labels = frame["KeyDesign"].to_numpy(dtype=int)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("{} must contain both binary labels 0 and 1".format(project))
    if np.min(np.bincount(labels, minlength=2)) < 2:
        raise ValueError("{} needs at least two samples from each class".format(project))

    id_column = next(
        (
            column
            for column in frame.columns
            if column.lower() in {"id", "node_id", "nodeid"}
        ),
        None,
    )
    node_ids = (
        frame[id_column].to_numpy()
        if id_column is not None
        else np.arange(len(frame), dtype=int)
    )
    return labels, node_ids


def repeated_two_fold_splits(labels, repeats, base_seed):

    for repeat in range(repeats):
        split_seed = base_seed + repeat
        splitter = StratifiedKFold(
            n_splits=2,
            shuffle=True,
            random_state=split_seed,
        )
        for fold, (train_idx, test_idx) in enumerate(
            splitter.split(np.zeros(len(labels)), labels)
        ):
            model_seed = (base_seed + repeat * 2 + fold) % (2**31 - 1)
            yield repeat, fold, split_seed, model_seed, train_idx, test_idx


def resample_training_fold(features, labels, seed):

    positive_ratio = float(np.sum(labels == 1)) / len(labels)
    class_counts = np.bincount(labels, minlength=2)
    minority_count = int(np.min(class_counts))
    if positive_ratio > 0.4 or minority_count < 2:
        return features, labels

    sampler = SMOTE(
        k_neighbors=min(2, minority_count - 1),
        random_state=seed,
    )
    return sampler.fit_resample(features, labels)


def fit_random_forest(features, labels, seed, grid_search=False, n_jobs=-1):
    estimator = RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        random_state=seed,
        n_jobs=n_jobs,
    )
    if not grid_search:
        estimator.fit(features, labels)
        return estimator, {}

    minimum_class_count = int(np.min(np.bincount(labels, minlength=2)))
    inner_folds = min(3, minimum_class_count)
    if inner_folds < 2:
        estimator.fit(features, labels)
        return estimator, {}

    inner_cv = StratifiedKFold(
        n_splits=inner_folds,
        shuffle=True,
        random_state=seed,
    )
    search = GridSearchCV(
        estimator,
        RF_PARAMETER_GRID,
        scoring="f1",
        cv=inner_cv,
        n_jobs=n_jobs,
    )
    search.fit(features, labels)
    return search.best_estimator_, search.best_params_


def evaluate_predictions(labels, predictions, probabilities):
    if np.unique(labels).size == 2:
        auc = roc_auc_score(labels, probabilities)
    else:
        auc = np.nan
    return {
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "auc": auc,
        "accuracy": accuracy_score(labels, predictions),
        "mcc": matthews_corrcoef(labels, predictions),
        "brier_score": brier_score_loss(labels, probabilities),
    }


def write_project_outputs(project_dir, fold_rows, split_rows):
    project_dir.mkdir(parents=True, exist_ok=True)
    fold_frame = pd.DataFrame(fold_rows)
    fold_frame.to_csv(project_dir / "metrics_by_fold.csv", index=False)
    pd.DataFrame(split_rows).to_csv(project_dir / "split_manifest.csv", index=False)

    if fold_frame.empty:
        return None

    summary = {
        "project": fold_rows[0]["project"],
        "variant": fold_rows[0]["variant"],
        "folds": len(fold_rows),
        "repeats": int(fold_frame["repeat"].nunique()),
    }
    for metric in METRIC_COLUMNS:
        summary["median_{}".format(metric)] = round(
            float(fold_frame[metric].median()), 4
        )
        summary["mean_{}".format(metric)] = round(
            float(fold_frame[metric].mean()), 4
        )
    pd.DataFrame([summary]).to_csv(project_dir / "summary.csv", index=False)
    return summary


def run_project(project, args):
    labels, node_ids = load_project_labels(project, args.dataset_root)
    fold_rows = []
    split_rows = []
    project_dir = args.output_dir / args.variant / project

    for repeat, fold, split_seed, model_seed, train_idx, test_idx in (
        repeated_two_fold_splits(labels, args.repeats, args.seed)
    ):
        print(
            "{}: repeat {}/{}, fold {}/2".format(
                project, repeat + 1, args.repeats, fold + 1
            )
        )


        fold_features = generate_fold_features(
            project=project,
            train_idx=train_idx,
            test_idx=test_idx,
            seed=model_seed,
            dataset_root=args.dataset_root,
            variant=args.variant,
            epochs=args.epochs,
            patience=args.patience,
        )
        if not np.array_equal(fold_features.train_idx, train_idx):
            raise AssertionError("Representation training indices changed")
        if not np.array_equal(fold_features.test_idx, test_idx):
            raise AssertionError("Representation test indices changed")

        y_train = labels[train_idx]
        y_test = labels[test_idx]
        x_train, y_resampled = resample_training_fold(
            fold_features.train_features,
            y_train,
            model_seed,
        )
        classifier, best_parameters = fit_random_forest(
            x_train,
            y_resampled,
            model_seed,
            grid_search=args.grid_search,
            n_jobs=args.n_jobs,
        )
        predictions = classifier.predict(fold_features.test_features)
        probabilities = classifier.predict_proba(fold_features.test_features)[:, 1]

        fold_row = {
            "project": project,
            "variant": args.variant,
            "repeat": repeat + 1,
            "fold": fold + 1,
            "split_seed": split_seed,
            "model_seed": model_seed,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "best_rf_parameters": json.dumps(best_parameters, sort_keys=True),
        }
        fold_row.update(evaluate_predictions(y_test, predictions, probabilities))
        fold_rows.append(fold_row)
        split_rows.append(
            {
                "project": project,
                "repeat": repeat + 1,
                "fold": fold + 1,
                "split_seed": split_seed,
                "model_seed": model_seed,
                "train_indices": json.dumps(train_idx.tolist()),
                "test_indices": json.dumps(test_idx.tolist()),
                "train_node_ids": json.dumps(node_ids[train_idx].tolist()),
                "test_node_ids": json.dumps(node_ids[test_idx].tolist()),
            }
        )
        write_project_outputs(project_dir, fold_rows, split_rows)

    return write_project_outputs(project_dir, fold_rows, split_rows)


def main():
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.patience < 1:
        raise ValueError("--patience must be at least 1")
    if not args.dataset_root.is_dir():
        raise FileNotFoundError("Dataset directory not found: {}".format(args.dataset_root))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [run_project(project, args) for project in resolve_projects(args)]
    pd.DataFrame(summaries).to_csv(
        args.output_dir / "{}_summary.csv".format(args.variant),
        index=False,
    )


if __name__ == "__main__":
    main()
