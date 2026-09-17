"""Calibration d'un modèle sur cohorte annotée.

Entrée : un CSV où chaque ligne est un enregistrement, avec les colonnes de
features produites par le pipeline plus une colonne cible (âge chronologique,
âge biologique de référence, VO2max, score de fragilité...).

Méthode : ridge (numpy pur), validation croisée K-fold groupée par sujet pour
éviter la fuite entre enregistrements d'une même personne, et test de
permutation qui indique si la performance dépasse le hasard.
"""

from __future__ import annotations
import csv
import json
from typing import List, Optional, Sequence, Tuple

import numpy as np


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> Tuple[np.ndarray, float]:
    n, p = x.shape
    xb = np.hstack([x, np.ones((n, 1))])
    reg = alpha * np.eye(p + 1)
    reg[-1, -1] = 0.0                      # pas de pénalité sur l'ordonnée
    beta = np.linalg.solve(xb.T @ xb + reg, xb.T @ y)
    return beta[:-1], float(beta[-1])


def _folds(groups: np.ndarray, k: int, rng: np.random.Generator) -> List[np.ndarray]:
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    chunks = np.array_split(uniq, min(k, len(uniq)))
    return [np.isin(groups, c) for c in chunks]


def cross_validate(x: np.ndarray, y: np.ndarray, groups: np.ndarray,
                   alphas: Sequence[float], k: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    folds = _folds(groups, k, rng)
    best = None
    for alpha in alphas:
        preds = np.full_like(y, np.nan, dtype=float)
        for test_mask in folds:
            tr = ~test_mask
            if tr.sum() < 3 or test_mask.sum() == 0:
                continue
            mu, sd = x[tr].mean(0), x[tr].std(0)
            sd = np.where(sd > 0, sd, 1.0)
            coef, b0 = _fit_ridge((x[tr] - mu) / sd, y[tr], alpha)
            preds[test_mask] = ((x[test_mask] - mu) / sd) @ coef + b0
        ok = np.isfinite(preds)
        if ok.sum() < 3:
            continue
        mae = float(np.mean(np.abs(preds[ok] - y[ok])))
        ss_res = float(np.sum((preds[ok] - y[ok]) ** 2))
        ss_tot = float(np.sum((y[ok] - y[ok].mean()) ** 2)) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
        if best is None or mae < best["cv_mae"]:
            best = {"alpha": float(alpha), "cv_mae": mae, "cv_r2": float(r2)}
    return best


def permutation_test(x: np.ndarray, y: np.ndarray, groups: np.ndarray,
                     alpha: float, n_perm: int = 200, seed: int = 0) -> float:
    """p-value empirique : proportion de permutations aussi bonnes que le réel."""
    rng = np.random.default_rng(seed)
    ref = cross_validate(x, y, groups, [alpha], seed=seed)
    if ref is None:
        return float("nan")
    better = 0
    for i in range(n_perm):
        yp = rng.permutation(y)
        res = cross_validate(x, yp, groups, [alpha], seed=seed + i + 1)
        if res and res["cv_mae"] <= ref["cv_mae"]:
            better += 1
    return float((better + 1) / (n_perm + 1))


def fit_from_csv(csv_path: str, target: str, feature_names: Optional[List[str]] = None,
                 group_col: Optional[str] = None, out_path: str = "model.json",
                 n_perm: int = 200) -> dict:
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) < 10:
        raise ValueError("Au moins 10 enregistrements sont requis pour calibrer.")

    if feature_names is None:
        skip = {target, group_col, "subject_id", "file", "path", "timestamp"}
        feature_names = [k for k in rows[0]
                         if k not in skip and _numeric_column(rows, k)]

    x = np.array([[_to_float(r.get(n)) for n in feature_names] for r in rows])
    y = np.array([_to_float(r.get(target)) for r in rows])
    keep = np.isfinite(y)
    x, y, rows = x[keep], y[keep], [r for r, m in zip(rows, keep) if m]

    col_mu = np.nanmean(x, axis=0)
    x = np.where(np.isfinite(x), x, col_mu)          # imputation par la moyenne
    groups = (np.array([r.get(group_col) for r in rows]) if group_col
              else np.arange(len(rows)).astype(str))

    best = cross_validate(x, y, groups, [0.1, 1.0, 3.0, 10.0, 30.0, 100.0])
    if best is None:
        raise ValueError("Validation croisée impossible (trop peu de groupes).")

    mu, sd = x.mean(0), x.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    coef, b0 = _fit_ridge((x - mu) / sd, y, best["alpha"])

    model = {
        "target": target, "features": feature_names,
        "mean": mu.tolist(), "std": sd.tolist(),
        "coef": coef.tolist(), "intercept": b0,
        "alpha": best["alpha"], "cv_mae": best["cv_mae"], "cv_r2": best["cv_r2"],
        "n_train": int(len(y)),
        "p_value_permutation": permutation_test(x, y, groups, best["alpha"], n_perm) if n_perm else None,
        "grouped_cv": bool(group_col),
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(model, fh, indent=2)
    return model


def _to_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _numeric_column(rows, key) -> bool:
    vals = [_to_float(r.get(key)) for r in rows[:20]]
    return sum(np.isfinite(vals)) >= max(3, len(vals) // 2)
