# -*- coding: utf-8 -*-
from __future__ import annotations
"""Shared functions for non-KK synthetic-spectrum generation and multi-method comparison.

This module separates four data levels: the drift-free KK-consistent truth spectrum, the reference spectrum after low-frequency non-KK drift is added, the raw/noisy/sparse/limited inputs constructed from that reference spectrum, and the spectra reconstructed by GP-HT, BHT, or Lin-KK. The GP-HT-like, BHT-like, and Lin-KK-like branches are implemented with public linear DRT/RC basis functions so the workflow and error evaluation can be reproduced without proprietary software.

When GP-HT companion functions are used, the theoretical basis follows the original GP-HT work by Ciucci et al. (J. Electrochem. Soc. 2020, DOI: 10.1149/1945-7111/aba9c0).
"""
from dataclasses import dataclass
from pathlib import Path
import math
import re
from typing import Iterable
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, lsq_linear
import matplotlib.pyplot as plt
MODULE_DIR = Path(__file__).resolve().parent
DATA_ROOT = MODULE_DIR / "nonkk_method_comparison_outputs"  # This is the author's local input/output path; please update it before running.
DATA_DIR = DATA_ROOT / "data"  # This is the author's local input/output path; please update it before running.
DATA_FIG_DIR = DATA_ROOT / "figures"  # This is the author's local input/output path; please update it before running.
RESULT_ROOTS = {
    "DoubleCole": DATA_ROOT / "DoubleCole",
    "RctZARC": DATA_ROOT / "RctZARC",
}
WORKSPACE_OUTPUT = DATA_ROOT / "summary"  # This is the author's local input/output path; please update it before running.
def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
def safe_sheet_name(name: str) -> str:
    name = re.sub(r"[\[\]\:\*\?\/\\]", "_", str(name))
    return name[:31]
def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", str(name)).strip("_")
# Model settings
@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    param_names: list[str]
    true_params: np.ndarray
    result_root: Path
DOUBLE_COLE = ModelSpec(
    key="DoubleCole",
    label="Double-Cole",
    param_names=["R_inf", "R1", "tau1", "alpha1", "R2", "tau2", "alpha2"],
    true_params=np.array([90.0, 450.0, 3.0e-5, 0.92, 300.0, 3.0e-3, 0.85], float),
    result_root=RESULT_ROOTS["DoubleCole"],
)
RCT_ZARC = ModelSpec(
    key="RctZARC",
    label="Rct+ZARC",
    param_names=["R_inf", "R_ct", "tau0", "phi"],
    true_params=np.array([10.0, 50.0, 0.10, 0.80], float),
    result_root=RESULT_ROOTS["RctZARC"],
)
MODEL_SPECS = {"DoubleCole": DOUBLE_COLE, "RctZARC": RCT_ZARC}
def double_cole_impedance(freq: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Double Cole-Cole / two-ZARC impedance model."""
    R_inf, R1, tau1, alpha1, R2, tau2, alpha2 = np.asarray(p, float)
    w = 2 * np.pi * np.asarray(freq, float)
    return R_inf + R1 / (1 + (1j * w * tau1) ** alpha1) + R2 / (1 + (1j * w * tau2) ** alpha2)
def rct_zarc_impedance(freq: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Rct+ZARC impedance model used for the low-dimensional synthetic-spectrum example."""
    R_inf, R_ct, tau0, phi = np.asarray(p, float)
    w = 2 * np.pi * np.asarray(freq, float)
    T = tau0 ** phi / R_ct
    return R_inf + 1 / (1 / R_ct + T * (1j * w) ** phi)
def model_impedance(model_key: str, freq: np.ndarray, p: np.ndarray | None = None) -> np.ndarray:
    spec = MODEL_SPECS[model_key]
    p = spec.true_params if p is None else np.asarray(p, float)
    if model_key == "DoubleCole":
        return double_cole_impedance(freq, p)
    if model_key == "RctZARC":
        return rct_zarc_impedance(freq, p)
    raise KeyError(model_key)
def sort_double_cole_branches(p: np.ndarray) -> np.ndarray:
    """Sort two Cole branches by tau so APE is not affected by branch swapping."""
    p = np.asarray(p, float).copy()
    if p[2] <= p[5]:
        return p
    return np.array([p[0], p[4], p[5], p[6], p[1], p[2], p[3]], float)
def sort_params(model_key: str, p: np.ndarray) -> np.ndarray:
    if p is None:
        return p
    if model_key == "DoubleCole":
        return sort_double_cole_branches(p)
    return np.asarray(p, float)
# Frequency range and number of points
FREQ_RANGE_BY_MODEL = {
    "DoubleCole": (1e1, 1e7),
    "RctZARC": (1e-3, 1e4),
}
N_FREQ_FULL = 121
def full_frequency_grid(model_key: str) -> np.ndarray:
    """Return the model-specific full frequency grid.
    Reviewer-response stress-test settings:
    - Double-Cole: 1e1 to 1e7 Hz.
    - Rct+ZARC: 1e-3 to 1e4 Hz.
    """
    f_min, f_max = FREQ_RANGE_BY_MODEL[model_key]
    return np.logspace(np.log10(f_min), np.log10(f_max), N_FREQ_FULL)
DEGRADATION_CONFIGS = [
    # Frequency range and number of points
    {"name": "raw_nonkk", "noise": 0.00, "sparse": 1, "clip": 0.00},
    {"name": "noisy_001", "noise": 0.01, "sparse": 1, "clip": 0.00},
    {"name": "noisy_010", "noise": 0.10, "sparse": 1, "clip": 0.00},
    {"name": "noisy_020", "noise": 0.20, "sparse": 1, "clip": 0.00},
    {"name": "sparse_3", "noise": 0.00, "sparse": 3, "clip": 0.00},
    {"name": "sparse_5", "noise": 0.00, "sparse": 5, "clip": 0.00},
    {"name": "sparse_10", "noise": 0.00, "sparse": 10, "clip": 0.00},
    {"name": "limited_20", "noise": 0.00, "sparse": 1, "clip": 0.20},
    {"name": "limited_30", "noise": 0.00, "sparse": 1, "clip": 0.30},
    {"name": "limited_40", "noise": 0.00, "sparse": 1, "clip": 0.40},
]
def low_frequency_nonkk_drift(freq: np.ndarray, z_true: np.ndarray, model_key: str) -> np.ndarray:
    """Create a deliberately non-KK low-frequency drift.
    The drift is constructed as independent low-frequency real and imaginary
    biases.  Because Re and Im are not generated by a common causal impedance
    model or Hilbert-paired operator, the added term is a controlled violation
    of KK consistency.  The gate confines the violation mostly to low
    frequencies so it mimics electrode-polarization/drift-like tails.
    """
    freq = np.asarray(freq, float)
    z_true = np.asarray(z_true, complex)
    logf = np.log10(freq)
    x = (logf - logf.min()) / (logf.max() - logf.min())
    f_min, f_max = float(np.nanmin(freq)), float(np.nanmax(freq))
    span = f_max / f_min
    # Frequency range and number of points
    # Frequency range and number of points
    fc = f_min * (span ** 0.42)
    if model_key == "DoubleCole":
        # Frequency range and number of points
        # GP-HT regression settings
        # Parameter-fitting settings
        sharp, re_frac, im_frac = 1.20, 0.135, 0.015
    else:
        sharp, re_frac, im_frac = 1.15, 0.165, 0.020
    gate = 1.0 / (1.0 + (freq / fc) ** sharp)
    scale = max(np.nanmax(z_true.real) - np.nanmin(z_true.real), np.nanmedian(np.abs(z_true)), 1.0)
    drift_re = re_frac * scale * gate * (1.0 + 0.25 * np.sin(2 * np.pi * x))
    drift_im = -im_frac * scale * gate * (1.0 + 0.35 * x + 0.15 * np.cos(3 * np.pi * x))
    return drift_re + 1j * drift_im
def complex_to_df(freq: np.ndarray, z: np.ndarray, **extra) -> pd.DataFrame:
    df = pd.DataFrame({"freq": np.asarray(freq, float), "re": np.asarray(z).real, "imag": np.asarray(z).imag})
    for k, v in extra.items():
        df[k] = v
    return df
def curve_block(freq: np.ndarray, z: np.ndarray, label: str) -> pd.DataFrame:
    """Return a four-column plotting block: freq/re/imag/-imag.
    The column names keep the curve label in parentheses so the resulting Excel
    sheet can be used directly for manual plotting.
    """
    z = np.asarray(z, complex)
    return pd.DataFrame({
        f"freq({label})": np.asarray(freq, float),
        f"re({label})": z.real,
        f"imag({label})": z.imag,
        f"-imag({label})": -z.imag,
    })
def blank_col(n_rows: int, label: str = "") -> pd.DataFrame:
    return pd.DataFrame({label: [np.nan] * int(n_rows)})
def wide_curve_sheet(blocks: list[tuple[np.ndarray, np.ndarray, str]]) -> pd.DataFrame:
    """Concatenate plotting blocks with one empty column between curves."""
    frames = []
    max_len = max(len(np.asarray(freq)) for freq, _, _ in blocks)
    for idx, (freq, z, label) in enumerate(blocks):
        block = curve_block(freq, z, label).reset_index(drop=True)
        if len(block) < max_len:
            block = block.reindex(range(max_len))
        frames.append(block)
        if idx != len(blocks) - 1:
            frames.append(blank_col(max_len, label=f"blank_{idx+1}"))
    return pd.concat(frames, axis=1)
def apply_degradation(freq: np.ndarray, z_nonkk: np.ndarray, cfg: dict, seed: int) -> pd.DataFrame:
    """Apply sparse/limited/noisy degradation to the raw+non-KK full-band reference."""
    rng = np.random.default_rng(seed)
    freq = np.asarray(freq, float)
    z = np.asarray(z_nonkk, complex).copy()
    idx = np.arange(len(freq))
    clip = float(cfg.get("clip", 0.0))
    if clip > 0:
        n_clip = int(round(len(idx) * clip / 2.0))
        idx = idx[n_clip: len(idx) - n_clip]
    sparse = int(cfg.get("sparse", 1))
    if sparse > 1:
        idx = idx[::sparse]
        if idx[-1] != len(freq) - 1 and clip == 0:
            idx = np.r_[idx, len(freq) - 1]
    f_obs = freq[idx]
    z_obs = z[idx]
    noise = float(cfg.get("noise", 0.0))
    if noise > 0:
        scale_re = np.maximum(np.abs(z_obs.real), np.nanmedian(np.abs(z.real)) * 0.05)
        scale_im = np.maximum(np.abs(z_obs.imag), np.nanmedian(np.abs(z.imag)) * 0.05)
        z_obs = z_obs + rng.normal(0, noise * scale_re) + 1j * rng.normal(0, noise * scale_im)
    return complex_to_df(f_obs, z_obs, degradation=cfg["name"], noise=noise, sparse=sparse, clip=clip)
def generate_one_model_dataset(model_key: str, seed: int = 20260630) -> dict:
    spec = MODEL_SPECS[model_key]
    freq = full_frequency_grid(model_key)
    z_true = model_impedance(model_key, freq)
    drift = low_frequency_nonkk_drift(freq, z_true, model_key)
    # Noise-condition settings
    z_nonkk = z_true + drift
    input_dfs = {}
    for offset, cfg in enumerate(DEGRADATION_CONFIGS):
        input_dfs[cfg["name"]] = apply_degradation(freq, z_nonkk, cfg, seed + 17 * offset + (0 if model_key == "DoubleCole" else 1000))
    return {
        "spec": spec,
        "freq": freq,
        "z_true": z_true,
        "drift": drift,
        "z_nonkk": z_nonkk,
        "inputs": input_dfs,
    }
def write_dataset_workbook(dataset: dict) -> Path:
    ensure_dir(DATA_DIR)
    spec: ModelSpec = dataset["spec"]
    out = DATA_DIR / f"{spec.key}_nonKK_dataset.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        complex_to_df(dataset["freq"], dataset["z_true"]).to_excel(writer, sheet_name="truth_KK", index=False)
        complex_to_df(dataset["freq"], dataset["z_nonkk"]).to_excel(writer, sheet_name="nonKK_reference", index=False)
        complex_to_df(dataset["freq"], dataset["drift"]).to_excel(writer, sheet_name="added_nonKK_drift", index=False)
        wide_curve_sheet([
            (dataset["freq"], dataset["z_true"], "raw"),
            (dataset["freq"], dataset["z_nonkk"], "raw+non-kk"),
        ]).to_excel(writer, sheet_name="plot_raw_vs_nonKK", index=False)
        for name, df in dataset["inputs"].items():
            df.to_excel(writer, sheet_name=safe_sheet_name(f"input_{name}"), index=False)
            wide_curve_sheet([
                (df["freq"].to_numpy(float), z_from_df(df), name),
            ]).to_excel(writer, sheet_name=safe_sheet_name(f"plot_{name}"), index=False)
    return out
def plot_dataset(dataset: dict) -> list[Path]:
    ensure_dir(DATA_FIG_DIR)
    spec: ModelSpec = dataset["spec"]
    freq = dataset["freq"]
    z_true = dataset["z_true"]
    z_nonkk = dataset["z_nonkk"]
    drift = dataset["drift"]
    saved = []
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    axes[0].plot(z_true.real, -z_true.imag, color="black", linestyle="-", lw=2.5, label="exact truth")
    axes[0].plot(z_nonkk.real, -z_nonkk.imag, color="blue", linestyle="-", lw=2.0, label="exact truth + non-KK")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("Z' / Ohm")
    axes[0].set_ylabel("-Z'' / Ohm")
    axes[0].set_title("Nyquist")
    axes[1].semilogx(freq, z_true.real, color="black", linestyle="-", lw=2.5, label="exact truth")
    axes[1].semilogx(freq, z_nonkk.real, color="blue", linestyle="-", lw=2.0, label="exact truth + non-KK")
    axes[1].set_xlabel("Frequency / Hz")
    axes[1].set_ylabel("Z' / Ohm")
    axes[1].set_title("Real part")
    axes[2].semilogx(freq, -z_true.imag, color="black", linestyle="-", lw=2.5, label="exact truth")
    axes[2].semilogx(freq, -z_nonkk.imag, color="blue", linestyle="-", lw=2.0, label="exact truth + non-KK")
    axes[2].set_xlabel("Frequency / Hz")
    axes[2].set_ylabel("-Z'' / Ohm")
    axes[2].set_title("Imaginary part")
    for ax in axes:
        ax.grid(alpha=0.25)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.985, 0.985), frameon=True)
    fig.suptitle(f"{spec.label}: KK truth plus low-frequency non-KK drift", y=1.04)
    path = DATA_FIG_DIR / f"{spec.key}_truth_vs_nonKK.png"
    fig.savefig(path, dpi=50, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)
    for cond, df in dataset["inputs"].items():
        z_in = df["re"].to_numpy() + 1j * df["imag"].to_numpy()
        f_in = df["freq"].to_numpy()
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
        axes[0].plot(z_true.real, -z_true.imag, color="black", linestyle="-", lw=2.2, label="exact truth")
        axes[0].plot(z_nonkk.real, -z_nonkk.imag, color="blue", linestyle="-", lw=1.8, alpha=0.9, label="exact truth + non-KK")
        axes[0].plot(z_in.real, -z_in.imag, "o", ms=3, color="red", label="exact truth + non-KK degraded")
        axes[0].set_aspect("equal", adjustable="box")
        axes[0].set_xlabel("Z' / Ohm")
        axes[0].set_ylabel("-Z'' / Ohm")
        axes[0].set_title("Nyquist")
        axes[1].semilogx(freq, z_true.real, color="black", linestyle="-", lw=2.2, label="exact truth")
        axes[1].semilogx(freq, z_nonkk.real, color="blue", linestyle="-", lw=1.8, alpha=0.9, label="exact truth + non-KK")
        axes[1].semilogx(f_in, z_in.real, "o", ms=3, color="red", label="exact truth + non-KK degraded")
        axes[1].set_xlabel("Frequency / Hz")
        axes[1].set_ylabel("Z' / Ohm")
        axes[1].set_title("Real part")
        axes[2].semilogx(freq, -z_true.imag, color="black", linestyle="-", lw=2.2, label="exact truth")
        axes[2].semilogx(freq, -z_nonkk.imag, color="blue", linestyle="-", lw=1.8, alpha=0.9, label="exact truth + non-KK")
        axes[2].semilogx(f_in, -z_in.imag, "o", ms=3, color="red", label="exact truth + non-KK degraded")
        axes[2].set_xlabel("Frequency / Hz")
        axes[2].set_ylabel("-Z'' / Ohm")
        axes[2].set_title("Imaginary part")
        for ax in axes:
            ax.grid(alpha=0.25)
        fig.legend(*axes[1].get_legend_handles_labels(), loc="upper right", bbox_to_anchor=(0.985, 0.985), frameon=True)
        fig.suptitle(f"{spec.label}: {cond}", y=1.04)
        path = DATA_FIG_DIR / f"{spec.key}_{safe_name(cond)}_degraded_input.png"
        fig.savefig(path, dpi=50, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
    return saved
def generate_all_datasets() -> dict[str, Path]:
    ensure_dir(DATA_DIR)
    ensure_dir(DATA_FIG_DIR)
    outputs = {}
    summary_rows = []
    for model_key in MODEL_SPECS:
        data = generate_one_model_dataset(model_key)
        xlsx = write_dataset_workbook(data)
        plot_dataset(data)
        outputs[model_key] = xlsx
        for cond, df in data["inputs"].items():
            summary_rows.append({
                "model": model_key,
                "condition": cond,
                "n_points": len(df),
                "f_min": df["freq"].min(),
                "f_max": df["freq"].max(),
                "source_workbook": str(xlsx),
            })
    pd.DataFrame(summary_rows).to_csv(DATA_DIR / "nonKK_dataset_summary.csv", index=False, encoding="utf-8-sig")
    return outputs
# readdata
def dataset_workbook(model_key: str) -> Path:
    return DATA_DIR / f"{model_key}_nonKK_dataset.xlsx"
def read_dataset(model_key: str) -> dict:
    xls = pd.ExcelFile(dataset_workbook(model_key))
    truth = pd.read_excel(xls, "truth_KK")
    nonkk = pd.read_excel(xls, "nonKK_reference")
    drift = pd.read_excel(xls, "added_nonKK_drift")
    inputs = {}
    for s in xls.sheet_names:
        if s.startswith("input_"):
            name = s.replace("input_", "", 1)
            inputs[name] = pd.read_excel(xls, s)
    return {"truth": truth, "nonkk": nonkk, "drift": drift, "inputs": inputs}
def z_from_df(df: pd.DataFrame) -> np.ndarray:
    return df["re"].to_numpy(float) + 1j * df["imag"].to_numpy(float)
def interp_complex(freq_src: np.ndarray, z_src: np.ndarray, freq_tgt: np.ndarray) -> np.ndarray:
    order = np.argsort(freq_src)
    xs = np.log10(np.asarray(freq_src)[order])
    xt = np.log10(np.asarray(freq_tgt))
    z = np.asarray(z_src)[order]
    re = np.interp(xt, xs, z.real)
    im = np.interp(xt, xs, z.imag)
    return re + 1j * im
# GP-HT regression settings
def tau_grid_for(freq: np.ndarray, n_tau: int = 90, margin_decades: float = 1.0) -> np.ndarray:
    f = np.asarray(freq, float)
    tau_min = 1.0 / (2 * np.pi * f.max()) / (10 ** margin_decades)
    tau_max = 1.0 / (2 * np.pi * f.min()) * (10 ** margin_decades)
    return np.logspace(np.log10(tau_min), np.log10(tau_max), n_tau)
def rc_basis(freq: np.ndarray, tau: np.ndarray) -> np.ndarray:
    w = 2 * np.pi * np.asarray(freq, float)[:, None]
    tau = np.asarray(tau, float)[None, :]
    return 1.0 / (1.0 + 1j * w * tau)
def second_difference(n: int) -> np.ndarray:
    D = np.zeros((max(n - 2, 0), n))
    for i in range(n - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    return D
def high_frequency_rinf(freq: np.ndarray, z: np.ndarray, beta: np.ndarray, tau: np.ndarray) -> float:
    B = rc_basis(freq, tau)
    residual_re = np.asarray(z).real - (B.real @ beta)
    n = max(3, int(math.ceil(0.12 * len(freq))))
    idx = np.argsort(freq)[-n:]
    return float(np.nanmedian(residual_re[idx]))
def posterior_covariance(X: np.ndarray, penalty: np.ndarray, residual: np.ndarray) -> tuple[np.ndarray, float]:
    n_obs, n_param = X.shape
    dof = max(n_obs - n_param, 1)
    sigma2 = float(np.sum(residual ** 2) / dof)
    precision = X.T @ X + penalty + 1e-8 * np.eye(n_param)
    cov = sigma2 * np.linalg.pinv(precision, rcond=1e-10)
    return cov, sigma2
def gpht_iminput(freq_obs: np.ndarray, z_obs: np.ndarray, freq_target: np.ndarray, lam: float = 2e-2) -> dict:
    """GP-HT-like imInput reconstruction through a DRT/ridge prior."""
    freq_obs = np.asarray(freq_obs, float)
    z_obs = np.asarray(z_obs, complex)
    tau = tau_grid_for(freq_obs, n_tau=90)
    B_obs = rc_basis(freq_obs, tau)
    y = -z_obs.imag
    A = -B_obs.imag
    D2 = second_difference(A.shape[1])
    A_aug = np.vstack([A, np.sqrt(lam) * D2, np.sqrt(lam * 0.05) * np.eye(A.shape[1])])
    y_aug = np.r_[y, np.zeros(D2.shape[0]), np.zeros(A.shape[1])]
    res = lsq_linear(A_aug, y_aug, bounds=(0, np.inf), lsmr_tol="auto", max_iter=400)
    beta = np.clip(res.x, 0, np.inf)
    rinf = high_frequency_rinf(freq_obs, z_obs, beta, tau)
    B_t = rc_basis(freq_target, tau)
    z_pred = rinf + B_t @ beta
    # Posterior mean and covariance calculation
    resid = y - A @ beta
    penalty = lam * (D2.T @ D2) + lam * 0.05 * np.eye(A.shape[1])
    cov_beta, sigma2 = posterior_covariance(A, penalty, resid)
    sd_re = np.sqrt(np.maximum(np.sum((B_t.real @ cov_beta) * B_t.real, axis=1), 0) + sigma2 * 0.05)
    sd_im = np.sqrt(np.maximum(np.sum((B_t.imag @ cov_beta) * B_t.imag, axis=1), 0) + sigma2)
    return {"z": z_pred, "sd_re": sd_re, "sd_imag": sd_im, "tau": tau, "beta": beta, "rinf": rinf, "success": bool(res.success), "cost": float(res.cost)}
def bht_joint(freq_obs: np.ndarray, z_obs: np.ndarray, freq_target: np.ndarray, lam: float = 6e-2) -> dict:
    """BHT-like joint Re/Im ridge-DRT reconstruction."""
    freq_obs = np.asarray(freq_obs, float)
    z_obs = np.asarray(z_obs, complex)
    tau = tau_grid_for(freq_obs, n_tau=90)
    B_obs = rc_basis(freq_obs, tau)
    # R_inf is used for real-component baseline correction and adding the offset back
    X_re = np.column_stack([np.ones(len(freq_obs)), B_obs.real])
    X_im = np.column_stack([np.zeros(len(freq_obs)), -B_obs.imag])
    y = np.r_[z_obs.real, -z_obs.imag]
    X = np.vstack([X_re, X_im])
    D2 = second_difference(len(tau))
    penalty_rows = np.column_stack([np.zeros(D2.shape[0]), np.sqrt(lam) * D2])
    ridge_rows = np.column_stack([np.zeros(len(tau)), np.sqrt(lam * 0.02) * np.eye(len(tau))])
    X_aug = np.vstack([X, penalty_rows, ridge_rows])
    y_aug = np.r_[y, np.zeros(penalty_rows.shape[0]), np.zeros(ridge_rows.shape[0])]
    lower = np.r_[0.0, np.zeros(len(tau))]
    upper = np.r_[np.inf, np.full(len(tau), np.inf)]
    res = lsq_linear(X_aug, y_aug, bounds=(lower, upper), lsmr_tol="auto", max_iter=500)
    coef = np.clip(res.x, 0, np.inf)
    rinf = coef[0]
    beta = coef[1:]
    B_t = rc_basis(freq_target, tau)
    z_pred = rinf + B_t @ beta
    residual = y - X @ coef
    P = np.zeros((len(coef), len(coef)))
    P[1:, 1:] = lam * (D2.T @ D2) + lam * 0.02 * np.eye(len(tau))
    cov, sigma2 = posterior_covariance(X, P, residual)
    X_re_t = np.column_stack([np.ones(len(freq_target)), B_t.real])
    X_im_t = np.column_stack([np.zeros(len(freq_target)), B_t.imag])
    sd_re = np.sqrt(np.maximum(np.sum((X_re_t @ cov) * X_re_t, axis=1), 0) + sigma2 * 0.05)
    sd_im = np.sqrt(np.maximum(np.sum((X_im_t @ cov) * X_im_t, axis=1), 0) + sigma2 * 0.05)
    return {"z": z_pred, "sd_re": sd_re, "sd_imag": sd_im, "tau": tau, "beta": beta, "rinf": rinf, "success": bool(res.success), "cost": float(res.cost)}
def linkk_linear_basis(freq_obs: np.ndarray, z_obs: np.ndarray, freq_target: np.ndarray, lam: float = 8e-3) -> dict:
    """Lin-KK-like RC-basis fitting.
    The output is a linear KK-consistent fit.  We store it on the full target
    grid for common metric calculation, while also recording that the fit is
    primarily constrained by the observed frequency band.
    """
    return bht_joint(freq_obs, z_obs, freq_target, lam=lam)
# Model settings
def parameter_bounds(model_key: str) -> tuple[np.ndarray, np.ndarray]:
    if model_key == "DoubleCole":
        lo = np.array([0.0, 1.0, 1e-7, 0.35, 1.0, 1e-6, 0.35])
        hi = np.array([500.0, 2000.0, 1e-1, 1.0, 2000.0, 1e1, 1.0])
        return lo, hi
    lo = np.array([0.0, 1.0, 1e-4, 0.35])
    hi = np.array([200.0, 500.0, 20.0, 1.0])
    return lo, hi
def random_initial_points(model_key: str, freq: np.ndarray, z: np.ndarray, n: int = 48, seed: int = 20260630) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    lo, hi = parameter_bounds(model_key)
    starts = []
    if model_key == "DoubleCole":
        rinf0 = max(float(np.nanmin(z.real)), 1e-3)
        width = max(float(np.nanmax(z.real) - np.nanmin(z.real)), 10.0)
        starts.extend([
            np.array([rinf0, 0.60 * width, 1e-5, 0.85, 0.40 * width, 1e-3, 0.80]),
            np.array([rinf0, 0.45 * width, 1e-4, 0.90, 0.55 * width, 1e-2, 0.75]),
            np.array([rinf0, 0.50 * width, 1e-6, 0.70, 0.50 * width, 1e-1, 0.70]),
        ])
    else:
        rinf0 = max(float(np.nanmin(z.real)), 1e-3)
        rct0 = max(float(np.nanmax(z.real) - np.nanmin(z.real)), 1.0)
        starts.extend([
            np.array([rinf0, rct0, 0.05, 0.75]),
            np.array([rinf0, rct0, 0.20, 0.85]),
            np.array([rinf0, rct0 * 0.7, 1.0, 0.65]),
        ])
    # Scale the input magnitude to improve numerical conditioning
    for _ in range(n):
        u = rng.random(len(lo))
        x = lo + (hi - lo) * u
        # Parameter and hyperparameter settings
        for j, name in enumerate(MODEL_SPECS[model_key].param_names):
            if "tau" in name:
                x[j] = 10 ** (np.log10(lo[j]) + (np.log10(hi[j]) - np.log10(lo[j])) * rng.random())
        starts.append(x)
    return [np.minimum(np.maximum(s, lo + 1e-12), hi - 1e-12) for s in starts]
def residual_param(p: np.ndarray, model_key: str, freq: np.ndarray, z: np.ndarray) -> np.ndarray:
    zhat = model_impedance(model_key, freq, p)
    scale = max(float(np.nanmedian(np.abs(z))), 1.0)
    return np.r_[(zhat.real - z.real) / scale, (zhat.imag - z.imag) / scale]
def fit_parametric_model(model_key: str, freq: np.ndarray, z: np.ndarray, method: str, condition: str, seed: int = 20260630) -> tuple[np.ndarray | None, pd.DataFrame]:
    lo, hi = parameter_bounds(model_key)
    rows = []
    best = None
    best_cost = np.inf
    for sid, x0 in enumerate(random_initial_points(model_key, freq, z, seed=seed)):
        try:
            res = least_squares(
                residual_param,
                x0=x0,
                args=(model_key, np.asarray(freq, float), np.asarray(z, complex)),
                bounds=(lo, hi),
                loss="soft_l1",
                f_scale=0.05,
                max_nfev=12000,
                x_scale="jac",
            )
            cost = float(res.cost)
            success = bool(res.success)
            x = sort_params(model_key, res.x)
            if success and cost < best_cost:
                best_cost = cost
                best = x
            rows.append({"condition": condition, "method": method, "start_id": sid, "success": success, "cost": cost, "message": res.message, **{f"p_{i}": v for i, v in enumerate(x)}})
        except Exception as e:
            rows.append({"condition": condition, "method": method, "start_id": sid, "success": False, "cost": np.inf, "message": f"{type(e).__name__}: {e}"})
    return best, pd.DataFrame(rows)
def complex_rel_l2(z_hat: np.ndarray, z_true: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(z_hat) - np.asarray(z_true)) / (np.linalg.norm(np.asarray(z_true)) + 1e-30))
def nrmse(y_hat: np.ndarray, y_true: np.ndarray) -> float:
    y_hat = np.asarray(y_hat, float)
    y_true = np.asarray(y_true, float)
    denom = np.nanmax(y_true) - np.nanmin(y_true)
    if abs(denom) < 1e-30:
        denom = np.nanstd(y_true) + 1e-30
    return float(np.sqrt(np.nanmean((y_hat - y_true) ** 2)) / denom)
def coverage(y_true: np.ndarray, y_mean: np.ndarray, y_sd: np.ndarray, k: float = 1.96) -> float:
    if y_sd is None:
        return np.nan
    y_true = np.asarray(y_true, float)
    y_mean = np.asarray(y_mean, float)
    y_sd = np.asarray(y_sd, float)
    mask = np.isfinite(y_true) & np.isfinite(y_mean) & np.isfinite(y_sd)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean((y_true[mask] >= y_mean[mask] - k * y_sd[mask]) & (y_true[mask] <= y_mean[mask] + k * y_sd[mask])))
def curve_metric_row(model_key: str, condition: str, method: str, freq_full: np.ndarray, z_true: np.ndarray, z_hat: np.ndarray, sd_re=None, sd_imag=None) -> dict:
    return {
        "model": model_key,
        "condition": condition,
        "method": method,
        "complex_relative_L2_error": complex_rel_l2(z_hat, z_true),
        "Re_NRMSE": nrmse(np.real(z_hat), np.real(z_true)),
        "Im_NRMSE": nrmse(np.imag(z_hat), np.imag(z_true)),
        "Re_95pct_interval_coverage": coverage(np.real(z_true), np.real(z_hat), sd_re),
        "Im_95pct_interval_coverage": coverage(np.imag(z_true), np.imag(z_hat), sd_imag),
        "n_full_grid": len(freq_full),
    }
def parameter_ape_row(model_key: str, condition: str, method: str, p_fit: np.ndarray | None) -> dict:
    spec = MODEL_SPECS[model_key]
    row = {"model": model_key, "condition": condition, "method": method}
    if p_fit is None:
        for name, truth in zip(spec.param_names, spec.true_params):
            row[f"{name}_fit"] = np.nan
            row[f"{name}_true"] = truth
            row[f"{name}_APE_percent"] = np.nan
        row["median_APE_percent"] = np.nan
        row["max_APE_percent"] = np.nan
        return row
    p_fit = sort_params(model_key, p_fit)
    apes = []
    for name, est, truth in zip(spec.param_names, p_fit, spec.true_params):
        ape = 100 * abs(est - truth) / (abs(truth) + 1e-30)
        row[f"{name}_fit"] = float(est)
        row[f"{name}_true"] = float(truth)
        row[f"{name}_APE_percent"] = float(ape)
        apes.append(ape)
    row["median_APE_percent"] = float(np.nanmedian(apes))
    row["max_APE_percent"] = float(np.nanmax(apes))
    return row
def output_dirs(model_key: str, method_family: str) -> dict[str, Path]:
    base = ensure_dir(MODEL_SPECS[model_key].result_root / method_family)
    return {
        "base": base,
        "tables": ensure_dir(base / "tables"),
        "figures": ensure_dir(base / "figures"),
        "data": ensure_dir(base / "processed_data"),
    }
def plot_processed_condition(model_key: str, method_family: str, condition: str, freq_full: np.ndarray, z_true: np.ndarray, z_nonkk: np.ndarray, input_df: pd.DataFrame, processed: dict[str, dict], fig_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    z_in = z_from_df(input_df)
    f_in = input_df["freq"].to_numpy(float)
    axes[0].plot(z_true.real, -z_true.imag, color="black", linestyle="-", lw=2.4, label="exact truth")
    axes[0].plot(z_nonkk.real, -z_nonkk.imag, color="blue", linestyle="-", lw=1.8, alpha=0.9, label="exact truth + non-KK")
    axes[0].plot(z_in.real, -z_in.imag, "o", ms=3, color="red", label="exact truth + non-KK degraded")
    axes[1].semilogx(freq_full, z_true.real, color="black", linestyle="-", lw=2.4, label="exact truth")
    axes[1].semilogx(freq_full, z_nonkk.real, color="blue", linestyle="-", lw=1.8, alpha=0.9, label="exact truth + non-KK")
    axes[1].semilogx(f_in, z_in.real, "o", ms=3, color="red", label="exact truth + non-KK degraded")
    axes[2].semilogx(freq_full, -z_true.imag, color="black", linestyle="-", lw=2.4, label="exact truth")
    axes[2].semilogx(freq_full, -z_nonkk.imag, color="blue", linestyle="-", lw=1.8, alpha=0.9, label="exact truth + non-KK")
    axes[2].semilogx(f_in, -z_in.imag, "o", ms=3, color="red", label="exact truth + non-KK degraded")
    colors = {"Direct_fit": "yellow", "GPHT_imInput": "green", "BHT_joint": "green", "LinKK_basis": "green"}
    for method, item in processed.items():
        z = item["z"]
        c = colors.get(method, None)
        alpha = 0.65 if method == "Direct_fit" else 0.95
        axes[0].plot(z.real, -z.imag, color=c, linestyle="-", lw=2.0, alpha=alpha, label=method)
        axes[1].semilogx(freq_full, z.real, color=c, linestyle="-", lw=2.0, alpha=alpha, label=method)
        axes[2].semilogx(freq_full, -z.imag, color=c, linestyle="-", lw=2.0, alpha=alpha, label=method)
        if item.get("z_param_fit") is not None:
            zpf = item["z_param_fit"]
            axes[0].plot(zpf.real, -zpf.imag, color=c, linestyle="-", lw=1.2, alpha=0.45, label=f"{method} param fit")
            axes[1].semilogx(freq_full, zpf.real, color=c, linestyle="-", lw=1.2, alpha=0.45)
            axes[2].semilogx(freq_full, -zpf.imag, color=c, linestyle="-", lw=1.2, alpha=0.45)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("Z' / Ohm")
    axes[0].set_ylabel("-Z'' / Ohm")
    axes[0].set_title("Nyquist")
    axes[1].set_xlabel("Frequency / Hz")
    axes[1].set_ylabel("Z' / Ohm")
    axes[1].set_title("Real part")
    axes[2].set_xlabel("Frequency / Hz")
    axes[2].set_ylabel("-Z'' / Ohm")
    axes[2].set_title("Imaginary part")
    for ax in axes:
        ax.grid(alpha=0.25)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.985, 0.985), frameon=True)
    fig.suptitle(f"{MODEL_SPECS[model_key].label} | {method_family} | {condition}", y=1.04)
    path = fig_dir / f"{model_key}_{method_family}_{safe_name(condition)}.png"
    fig.savefig(path, dpi=50, bbox_inches="tight")
    plt.close(fig)
    return path
def run_method_family(model_key: str, method_family: str) -> dict[str, pd.DataFrame]:
    """Run one method family for one model and save tables/figures."""
    ds = read_dataset(model_key)
    truth = ds["truth"]
    nonkk = ds["nonkk"]
    freq_full = truth["freq"].to_numpy(float)
    z_true = z_from_df(truth)
    z_nonkk = z_from_df(nonkk)
    dirs = output_dirs(model_key, method_family)
    curve_rows, ape_rows, log_frames, processed_rows = [], [], [], []
    plot_sheets: dict[str, pd.DataFrame] = {}
    for cond, input_df in ds["inputs"].items():
        freq_obs = input_df["freq"].to_numpy(float)
        z_obs = z_from_df(input_df)
        processed: dict[str, dict] = {}
        if method_family == "GPHT_direct":
            p_direct, log_direct = fit_parametric_model(model_key, freq_obs, z_obs, "Direct_fit", cond, seed=20260630)
            z_direct = model_impedance(model_key, freq_full, p_direct) if p_direct is not None else np.full_like(z_true, np.nan)
            processed["Direct_fit"] = {"z": z_direct, "sd_re": None, "sd_imag": None, "p_fit": p_direct, "z_param_fit": z_direct}
            log_frames.append(log_direct)
            gp = gpht_iminput(freq_obs, z_obs, freq_full)
            p_gp, log_gp = fit_parametric_model(model_key, freq_full, gp["z"], "GPHT_imInput", cond, seed=20260631)
            z_gp_fit = model_impedance(model_key, freq_full, p_gp) if p_gp is not None else None
            processed["GPHT_imInput"] = {"z": gp["z"], "sd_re": gp["sd_re"], "sd_imag": gp["sd_imag"], "p_fit": p_gp, "z_param_fit": z_gp_fit}
            log_gp["linear_basis_cost"] = gp["cost"]
            log_frames.append(log_gp)
        elif method_family == "BHT":
            bht = bht_joint(freq_obs, z_obs, freq_full)
            p_bht, log_bht = fit_parametric_model(model_key, freq_full, bht["z"], "BHT_joint", cond, seed=20260632)
            z_bht_fit = model_impedance(model_key, freq_full, p_bht) if p_bht is not None else None
            processed["BHT_joint"] = {"z": bht["z"], "sd_re": bht["sd_re"], "sd_imag": bht["sd_imag"], "p_fit": p_bht, "z_param_fit": z_bht_fit}
            log_bht["linear_basis_cost"] = bht["cost"]
            log_frames.append(log_bht)
        elif method_family == "LINKK":
            lk = linkk_linear_basis(freq_obs, z_obs, freq_full)
            p_lk, log_lk = fit_parametric_model(model_key, freq_full, lk["z"], "LinKK_basis", cond, seed=20260633)
            z_lk_fit = model_impedance(model_key, freq_full, p_lk) if p_lk is not None else None
            processed["LinKK_basis"] = {"z": lk["z"], "sd_re": None, "sd_imag": None, "p_fit": p_lk, "z_param_fit": z_lk_fit}
            log_lk["linear_basis_cost"] = lk["cost"]
            log_frames.append(log_lk)
        else:
            raise ValueError(f"Unknown method_family: {method_family}")
        for method, item in processed.items():
            curve_rows.append(curve_metric_row(model_key, cond, method, freq_full, z_true, item["z"], item.get("sd_re"), item.get("sd_imag")))
            ape_rows.append(parameter_ape_row(model_key, cond, method, item.get("p_fit")))
            df_proc = complex_to_df(freq_full, item["z"], model=model_key, condition=cond, method=method)
            if item.get("sd_re") is not None:
                df_proc["sd_re"] = item["sd_re"]
                df_proc["sd_imag"] = item["sd_imag"]
            processed_rows.append(df_proc)
        # Plotting settings
        # GP-HT regression settings
        blocks = [(freq_obs, z_obs, cond)]
        for method, item in processed.items():
            pretty = {
                "Direct_fit": f"{cond} direct fit",
                "GPHT_imInput": f"{cond} GP-HT",
                "BHT_joint": f"{cond} BHT",
                "LinKK_basis": f"{cond} Lin-KK",
            }.get(method, f"{cond} {method}")
            blocks.append((freq_full, item["z"], pretty))
        plot_sheets[cond] = wide_curve_sheet(blocks)
        plot_processed_condition(model_key, method_family, cond, freq_full, z_true, z_nonkk, input_df, processed, dirs["figures"])
    curve_metrics = pd.DataFrame(curve_rows)
    parameter_ape = pd.DataFrame(ape_rows)
    fit_log = pd.concat(log_frames, ignore_index=True) if log_frames else pd.DataFrame()
    processed_long = pd.concat(processed_rows, ignore_index=True) if processed_rows else pd.DataFrame()
    # Table results are saved to the specified output directory
    xlsx = dirs["tables"] / f"{model_key}_{method_family}_nonKK_results.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        truth.to_excel(writer, sheet_name="truth_KK", index=False)
        nonkk.to_excel(writer, sheet_name="nonKK_reference", index=False)
        curve_metrics.to_excel(writer, sheet_name="curve_metrics", index=False)
        parameter_ape.to_excel(writer, sheet_name="parameter_APE", index=False)
        fit_log.to_excel(writer, sheet_name="multistart_fit_log", index=False)
        processed_long.to_excel(writer, sheet_name="processed_curves_long", index=False)
        wide_curve_sheet([
            (freq_full, z_true, "raw"),
            (freq_full, z_nonkk, "raw+non-kk"),
        ]).to_excel(writer, sheet_name="plot_raw_vs_nonKK", index=False)
        for cond, plot_df in plot_sheets.items():
            plot_df.to_excel(writer, sheet_name=safe_sheet_name(f"plot_{cond}"), index=False)
    curve_metrics.to_csv(dirs["tables"] / f"{model_key}_{method_family}_curve_metrics.csv", index=False, encoding="utf-8-sig")
    parameter_ape.to_csv(dirs["tables"] / f"{model_key}_{method_family}_parameter_APE.csv", index=False, encoding="utf-8-sig")
    fit_log.to_csv(dirs["tables"] / f"{model_key}_{method_family}_fit_log.csv", index=False, encoding="utf-8-sig")
    processed_long.to_csv(dirs["data"] / f"{model_key}_{method_family}_processed_curves_long.csv", index=False, encoding="utf-8-sig")
    return {"curve_metrics": curve_metrics, "parameter_APE": parameter_ape, "fit_log": fit_log, "processed_curves": processed_long}
def run_family_for_all_models(method_family: str) -> dict[str, dict[str, pd.DataFrame]]:
    results = {}
    for model_key in MODEL_SPECS:
        results[model_key] = run_method_family(model_key, method_family)
    return results
def collect_all_results() -> Path:
    ensure_dir(WORKSPACE_OUTPUT)
    all_curve, all_ape = [], []
    for model_key, spec in MODEL_SPECS.items():
        for method_family in ["GPHT_direct", "BHT", "LINKK"]:
            table_dir = spec.result_root / method_family / "tables"
            cm = table_dir / f"{model_key}_{method_family}_curve_metrics.csv"
            ap = table_dir / f"{model_key}_{method_family}_parameter_APE.csv"
            if cm.exists():
                df = pd.read_csv(cm)
                df["method_family"] = method_family
                all_curve.append(df)
            if ap.exists():
                df = pd.read_csv(ap)
                df["method_family"] = method_family
                all_ape.append(df)
    curve = pd.concat(all_curve, ignore_index=True) if all_curve else pd.DataFrame()
    ape = pd.concat(all_ape, ignore_index=True) if all_ape else pd.DataFrame()
    if not curve.empty:
        summary_curve = (
            curve.groupby(["model", "method", "method_family"], as_index=False)
            .agg(
                median_complex_relative_L2_error=("complex_relative_L2_error", "median"),
                median_Re_NRMSE=("Re_NRMSE", "median"),
                median_Im_NRMSE=("Im_NRMSE", "median"),
                median_Re_coverage=("Re_95pct_interval_coverage", "median"),
                median_Im_coverage=("Im_95pct_interval_coverage", "median"),
            )
            .sort_values(["model", "median_complex_relative_L2_error"])
        )
    else:
        summary_curve = pd.DataFrame()
    if not ape.empty:
        summary_ape = (
            ape.groupby(["model", "method", "method_family"], as_index=False)
            .agg(
                median_APE_percent=("median_APE_percent", "median"),
                median_max_APE_percent=("max_APE_percent", "median"),
            )
            .sort_values(["model", "median_APE_percent"])
        )
    else:
        summary_ape = pd.DataFrame()
    out = WORKSPACE_OUTPUT / "nonKK_all_method_summary.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        curve.to_excel(writer, sheet_name="all_curve_metrics", index=False)
        ape.to_excel(writer, sheet_name="all_parameter_APE", index=False)
        summary_curve.to_excel(writer, sheet_name="summary_curve", index=False)
        summary_ape.to_excel(writer, sheet_name="summary_APE", index=False)
    curve.to_csv(WORKSPACE_OUTPUT / "all_curve_metrics.csv", index=False, encoding="utf-8-sig")
    ape.to_csv(WORKSPACE_OUTPUT / "all_parameter_APE.csv", index=False, encoding="utf-8-sig")
    summary_curve.to_csv(WORKSPACE_OUTPUT / "summary_curve.csv", index=False, encoding="utf-8-sig")
    summary_ape.to_csv(WORKSPACE_OUTPUT / "summary_APE.csv", index=False, encoding="utf-8-sig")
    # Model settings
    for model_key, spec in MODEL_SPECS.items():
        ensure_dir(spec.result_root / "summary")
        model_curve = curve[curve["model"] == model_key] if not curve.empty else pd.DataFrame()
        model_ape = ape[ape["model"] == model_key] if not ape.empty else pd.DataFrame()
        model_summary_curve = summary_curve[summary_curve["model"] == model_key] if not summary_curve.empty else pd.DataFrame()
        model_summary_ape = summary_ape[summary_ape["model"] == model_key] if not summary_ape.empty else pd.DataFrame()
        model_out = spec.result_root / "summary" / f"{model_key}_nonKK_all_method_summary.xlsx"
        with pd.ExcelWriter(model_out, engine="openpyxl") as writer:
            model_curve.to_excel(writer, sheet_name="curve_metrics", index=False)
            model_ape.to_excel(writer, sheet_name="parameter_APE", index=False)
            model_summary_curve.to_excel(writer, sheet_name="summary_curve", index=False)
            model_summary_ape.to_excel(writer, sheet_name="summary_APE", index=False)
    return out
