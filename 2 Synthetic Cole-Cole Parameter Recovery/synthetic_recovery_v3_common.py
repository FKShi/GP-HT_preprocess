# -*- coding: utf-8 -*-
from __future__ import annotations
"""Shared functions for Cole-Cole parameter recovery from synthetic spectra.

This module reads synthetic spectra processed by different methods, refits the corresponding parametric model, and summarizes curve errors, parameter errors, and full-grid errors. The functions are shared by several notebooks, so their inputs and outputs are kept explicit for easier inspection.

When GP-HT companion functions are used, the theoretical basis follows the original GP-HT work by Ciucci et al. (J. Electrochem. Soc. 2020, DOI: 10.1149/1945-7111/aba9c0).
"""
from dataclasses import dataclass
from pathlib import Path
import math
import re
from typing import Callable, Iterable
import numpy as np
import pandas as pd
try:
    from scipy.optimize import least_squares
except ModuleNotFoundError:
    least_squares = None
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None
# Model settings
@dataclass(frozen=True)
class ModelSpec:
    """Container for one synthetic impedance model."""
    name: str
    param_names: list[str]
    true_params: np.ndarray
    model_func: Callable[[np.ndarray, np.ndarray], np.ndarray]
    default_output_folder: str
    notes: str = ""
def double_cole_impedance(freq: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Double-relaxation impedance model used for the double Cole-Cole example."""
    R_inf, R1, tau1, alpha1, R2, tau2, alpha2 = np.asarray(p, float)
    w = 2 * np.pi * np.asarray(freq, float)
    z = R_inf
    z = z + R1 / (1 + (1j * w * tau1) ** alpha1)
    z = z + R2 / (1 + (1j * w * tau2) ** alpha2)
    return z
def rct_zarc_impedance(freq: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Single-relaxation electrochemical impedance model used for the Rct+ZARC example."""
    R_inf, R_ct, tau0, phi = np.asarray(p, float)
    w = 2 * np.pi * np.asarray(freq, float)
    T = tau0 ** phi / R_ct
    return R_inf + 1 / (1 / R_ct + T * (1j * w) ** phi)
def two_zarc_warburg_impedance(freq: np.ndarray, p: np.ndarray) -> np.ndarray:
    """ ZARC branch Warburg diffusion tail.
theused fordiffusion tail of reconstructionerror.parameter APE as metricretain,
reconstructionerror as."""
    R_s, R1, tau1, alpha1, R2, tau2, alpha2, sigma_w = np.asarray(p, float)
    w = 2 * np.pi * np.asarray(freq, float)
    z = R_s
    z = z + R1 / (1 + (1j * w * tau1) ** alpha1)
    z = z + R2 / (1 + (1j * w * tau2) ** alpha2)
    z = z + sigma_w / np.sqrt(1j * w)
    return z
MODEL_SPECS = {
    "doubleCole": ModelSpec(
        name="doubleCole",
        param_names=["R_inf", "R1", "tau1", "alpha1", "R2", "tau2", "alpha2"],
        true_params=np.array([90.0, 450.0, 3.0e-5, 0.92, 300.0, 3.0e-3, 0.85], dtype=float),
        model_func=double_cole_impedance,
        default_output_folder="DoubleCole_recoveryV3",
        notes="Double-relaxation parameter-recovery model with known ground truth.",
    ),
    "RctZARC": ModelSpec(
        name="RctZARC",
        param_names=["R_inf", "R_ct", "tau0", "phi"],
        true_params=np.array([10.0, 50.0, 0.10, 0.80], dtype=float),
        model_func=rct_zarc_impedance,
        default_output_folder="RctZARC_recoveryV3",
        notes="Single-relaxation parameter-recovery model with known ground truth.",
    ),
    "TwoZARCWarburg": ModelSpec(
        name="TwoZARCWarburg",
        param_names=["R_s", "R1", "tau1", "alpha1", "R2", "tau2", "alpha2", "sigma_W"],
        true_params=np.array([8.0, 55.0, 2.0e-3, 0.86, 90.0, 3.5e-1, 0.78, 18.0], dtype=float),
        model_func=two_zarc_warburg_impedance,
        default_output_folder="TwoZARCWarburg_recoveryV3",
        notes="Diffusion-tail stress test; use parameter APE as diagnostic, not as the primary claim.",
    ),
}
def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
def safe_sheet_name(name: str) -> str:
    """Excel sheet names must be <=31 chars and cannot contain []:*?/\\."""
    name = re.sub(r"[\[\]\:\*\?\/\\]", "_", str(name))
    return name[:31]
def find_existing_path(*candidates: str | Path | None) -> Path:
    """Return the first existing path from a candidate list."""
    for p in candidates:
        if not p:
            continue
        path = Path(p)
        if path.exists():
            return path
    raise FileNotFoundError("None of the candidate paths exists:\n" + "\n".join(str(p) for p in candidates if p))
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common frequency/real/imaginary column names."""
    df = df.copy()
    df = df.rename(columns=lambda c: str(c).strip())
    alias = {
        "frequency": "freq",
        "Frequency": "freq",
        "freq_hz": "freq",
        "Freq": "freq",
        "f": "freq",
        "f_Hz": "freq",
        "real": "re",
        "Re": "re",
        "Zre": "re",
        "imaginary": "imag",
        "Im": "imag",
        "Zim": "imag",
        "imag_part": "imag",
    }
    return df.rename(columns={k: v for k, v in alias.items() if k in df.columns})
def z_from_df(df: pd.DataFrame, re_col: str = "re", im_col: str = "imag") -> np.ndarray:
    df = standardize_columns(df)
    return df[re_col].to_numpy(float) + 1j * df[im_col].to_numpy(float)
def clean_curve_df(df: pd.DataFrame, re_col: str = "re", im_col: str = "imag") -> pd.DataFrame:
    """Keep finite positive-frequency rows and sort by frequency."""
    df = standardize_columns(df)
    out = df[["freq", re_col, im_col]].copy()
    mask = np.isfinite(out["freq"]) & np.isfinite(out[re_col]) & np.isfinite(out[im_col]) & (out["freq"] > 0)
    out = out.loc[mask].sort_values("freq").reset_index(drop=True)
    out = out.rename(columns={re_col: "re", im_col: "imag"})
    return out
def interp_complex(freq_src: np.ndarray, z_src: np.ndarray, freq_tgt: np.ndarray) -> np.ndarray:
    """Interpolate a complex curve on log10-frequency.
    This is interpolation only.  The caller records whether the target grid
    extends outside the available source frequency range.
    """
    freq_src = np.asarray(freq_src, float)
    freq_tgt = np.asarray(freq_tgt, float)
    z_src = np.asarray(z_src, complex)
    order = np.argsort(freq_src)
    x_src = np.log10(freq_src[order])
    x_tgt = np.log10(freq_tgt)
    re = np.interp(x_tgt, x_src, z_src.real[order])
    im = np.interp(x_tgt, x_src, z_src.imag[order])
    return re + 1j * im
# Error-metric calculation
def complex_rel_l2(z_hat: np.ndarray, z_true: np.ndarray) -> float:
    """Relative L2 error for a complex spectrum: ||Zhat-Ztrue||2 / ||Ztrue||2."""
    z_hat = np.asarray(z_hat, complex)
    z_true = np.asarray(z_true, complex)
    return float(np.linalg.norm(z_hat - z_true) / (np.linalg.norm(z_true) + 1e-30))
def nrmse(y_hat: np.ndarray, y_true: np.ndarray) -> float:
    """Range-normalized RMSE for a real-valued component."""
    y_hat = np.asarray(y_hat, float)
    y_true = np.asarray(y_true, float)
    denom = np.nanmax(y_true) - np.nanmin(y_true)
    if not np.isfinite(denom) or abs(denom) < 1e-30:
        denom = np.nanstd(y_true) + 1e-30
    return float(np.sqrt(np.nanmean((y_hat - y_true) ** 2)) / denom)
def coverage_rate(y_true: np.ndarray, mean: np.ndarray, sd: np.ndarray, k: float = 1.96) -> float:
    """Pointwise posterior interval coverage rate."""
    y_true = np.asarray(y_true, float)
    mean = np.asarray(mean, float)
    sd = np.asarray(sd, float)
    mask = np.isfinite(y_true) & np.isfinite(mean) & np.isfinite(sd)
    if mask.sum() == 0:
        return np.nan
    lower = mean[mask] - k * sd[mask]
    upper = mean[mask] + k * sd[mask]
    return float(np.mean((y_true[mask] >= lower) & (y_true[mask] <= upper)))
def curve_metric_row(
    truth_df: pd.DataFrame,
    curve_df: pd.DataFrame,
    condition: str,
    method: str,
    sd_re_col: str | None = None,
    sd_im_col: str | None = None,
) -> dict:
    """Compute curve-recovery metrics on available grid and full truth grid.
    full_grid_* metrics are only filled when the curve covers the full truth
    frequency range.  This prevents Lin-KK or band-limited inputs from being
    silently evaluated by endpoint extrapolation.
    """
    truth_df = clean_curve_df(truth_df)
    curve_df = standardize_columns(curve_df).sort_values("freq").reset_index(drop=True)
    f_truth = truth_df["freq"].to_numpy(float)
    z_truth = z_from_df(truth_df)
    f_curve = curve_df["freq"].to_numpy(float)
    z_curve = curve_df["re"].to_numpy(float) + 1j * curve_df["imag"].to_numpy(float)
    z_truth_on_curve = interp_complex(f_truth, z_truth, f_curve)
    row = {
        "condition": condition,
        "method": method,
        "n_points": int(len(f_curve)),
        "freq_min": float(np.nanmin(f_curve)),
        "freq_max": float(np.nanmax(f_curve)),
        "available_grid_complex_rel_l2": complex_rel_l2(z_curve, z_truth_on_curve),
        "available_grid_nrmse_re": nrmse(z_curve.real, z_truth_on_curve.real),
        "available_grid_nrmse_imag": nrmse(z_curve.imag, z_truth_on_curve.imag),
    }
    covers_full = (np.nanmin(f_curve) <= np.nanmin(f_truth) * (1 + 1e-12)) and (
        np.nanmax(f_curve) >= np.nanmax(f_truth) * (1 - 1e-12)
    )
    row["covers_full_truth_frequency_range"] = bool(covers_full)
    if covers_full:
        z_curve_full = interp_complex(f_curve, z_curve, f_truth)
        row["full_grid_complex_rel_l2"] = complex_rel_l2(z_curve_full, z_truth)
        row["full_grid_nrmse_re"] = nrmse(z_curve_full.real, z_truth.real)
        row["full_grid_nrmse_imag"] = nrmse(z_curve_full.imag, z_truth.imag)
    else:
        row["full_grid_complex_rel_l2"] = np.nan
        row["full_grid_nrmse_re"] = np.nan
        row["full_grid_nrmse_imag"] = np.nan
    if sd_re_col and sd_re_col in curve_df.columns:
        sd_re = curve_df[sd_re_col].to_numpy(float)
        row["available_grid_coverage_re_95"] = coverage_rate(z_truth_on_curve.real, z_curve.real, sd_re)
    if sd_im_col and sd_im_col in curve_df.columns:
        sd_im = curve_df[sd_im_col].to_numpy(float)
        row["available_grid_coverage_imag_95"] = coverage_rate(z_truth_on_curve.imag, z_curve.imag, sd_im)
    return row
# Parameter-fitting settings
def sort_relaxation_branches(p: np.ndarray, spec: ModelSpec) -> np.ndarray:
    """Sort two relaxation branches by tau to avoid label switching."""
    p = np.asarray(p, float).copy()
    if spec.name == "doubleCole":
        b1 = p[[1, 2, 3]]
        b2 = p[[4, 5, 6]]
        if b1[1] > b2[1]:
            p[[1, 2, 3, 4, 5, 6]] = np.r_[b2, b1]
    elif spec.name == "TwoZARCWarburg":
        b1 = p[[1, 2, 3]]
        b2 = p[[4, 5, 6]]
        if b1[1] > b2[1]:
            p[[1, 2, 3, 4, 5, 6]] = np.r_[b2, b1]
    return p
def dynamic_bounds(freq: np.ndarray, z: np.ndarray, spec: ModelSpec) -> tuple[np.ndarray, np.ndarray]:
    """Broad data-adaptive bounds; not centered on the true parameters."""
    freq = np.asarray(freq, float)
    z = np.asarray(z, complex)
    re = z.real
    width = max(float(np.nanpercentile(re, 99) - np.nanpercentile(re, 1)), 1.0)
    r_max = max(float(np.nanmax(re)), 1.0)
    tau_min = max(1 / (2 * np.pi * np.nanmax(freq)) / 100, 1e-10)
    tau_max = min(1 / (2 * np.pi * np.nanmin(freq)) * 100, 1e5)
    if spec.name == "doubleCole":
        lower = np.array([0, 0, tau_min, 0.35, 0, tau_min, 0.35], float)
        upper = np.array([max(r_max * 2, 1), width * 5, tau_max, 0.999, width * 5, tau_max, 0.999], float)
    elif spec.name == "RctZARC":
        lower = np.array([0, 0, tau_min, 0.35], float)
        upper = np.array([max(r_max * 2, 1), width * 8, tau_max, 0.999], float)
    elif spec.name == "TwoZARCWarburg":
        lower = np.array([0, 0, tau_min, 0.35, 0, tau_min, 0.35, 0], float)
        upper = np.array([max(r_max * 2, 1), width * 6, tau_max, 0.999, width * 6, tau_max, 0.999, width * 5], float)
    else:
        raise ValueError(f"Unknown spec: {spec.name}")
    return lower, upper
def initial_points(freq: np.ndarray, z: np.ndarray, spec: ModelSpec, n_random: int = 96, seed: int = 20260627) -> list[np.ndarray]:
    """Generate deterministic + random initial guesses inside broad bounds."""
    rng = np.random.default_rng(seed)
    lower, upper = dynamic_bounds(freq, z, spec)
    re = z.real
    width = max(float(np.nanpercentile(re, 99) - np.nanpercentile(re, 1)), 1.0)
    starts: list[np.ndarray] = []
    # Scale the input magnitude to improve numerical conditioning
    if spec.name == "doubleCole":
        starts.append(np.array([max(np.nanmin(re), 0), 0.5 * width, spec.true_params[2], 0.85, 0.5 * width, spec.true_params[5], 0.85]))
    elif spec.name == "RctZARC":
        starts.append(np.array([max(np.nanmin(re), 0), max(width, 1.0), spec.true_params[2], 0.80]))
    elif spec.name == "TwoZARCWarburg":
        starts.append(np.array([max(np.nanmin(re), 0), 0.35 * width, spec.true_params[2], 0.85, 0.55 * width, spec.true_params[5], 0.80, max(1.0, 0.05 * width)]))
    for _ in range(n_random):
        if spec.name == "doubleCole":
            p = np.array([
                rng.uniform(lower[0], min(upper[0], max(np.nanmin(re) + width, 1.0))),
                rng.uniform(max(lower[1], 0.05 * width), min(upper[1], 2.5 * width)),
                10 ** rng.uniform(np.log10(lower[2]), np.log10(upper[2])),
                rng.uniform(0.45, 0.98),
                rng.uniform(max(lower[4], 0.05 * width), min(upper[4], 2.5 * width)),
                10 ** rng.uniform(np.log10(lower[5]), np.log10(upper[5])),
                rng.uniform(0.45, 0.98),
            ])
        elif spec.name == "RctZARC":
            p = np.array([
                rng.uniform(lower[0], min(upper[0], max(np.nanmin(re) + width, 1.0))),
                rng.uniform(max(lower[1], 0.05 * width), min(upper[1], 4.0 * width)),
                10 ** rng.uniform(np.log10(lower[2]), np.log10(upper[2])),
                rng.uniform(0.45, 0.98),
            ])
        else:
            p = np.array([
                rng.uniform(lower[0], min(upper[0], max(np.nanmin(re) + width, 1.0))),
                rng.uniform(max(lower[1], 0.03 * width), min(upper[1], 3.0 * width)),
                10 ** rng.uniform(np.log10(lower[2]), np.log10(upper[2])),
                rng.uniform(0.45, 0.98),
                rng.uniform(max(lower[4], 0.03 * width), min(upper[4], 3.0 * width)),
                10 ** rng.uniform(np.log10(lower[5]), np.log10(upper[5])),
                rng.uniform(0.45, 0.98),
                rng.uniform(max(lower[7], 0.005 * width), min(upper[7], 1.5 * width)),
            ])
        starts.append(p)
    out = []
    seen = set()
    for p in starts:
        p = sort_relaxation_branches(np.asarray(p, float), spec)
        p = np.minimum(np.maximum(p, lower + 1e-12), upper - 1e-12)
        key = tuple(np.round(np.log10(np.maximum(p, 1e-30)), 4))
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out
def residual_vector(p: np.ndarray, freq: np.ndarray, z_target: np.ndarray, spec: ModelSpec, rel_floor: float = 0.02) -> np.ndarray:
    """Modulus-relative complex residual used for bounded nonlinear fitting."""
    z_fit = spec.model_func(freq, p)
    mag_floor = rel_floor * np.nanmedian(np.abs(z_target) + 1e-30)
    scale = np.maximum(np.abs(z_target), mag_floor)
    r = (z_fit - z_target) / scale
    return np.r_[r.real, r.imag]
def fit_multistart(freq: np.ndarray, z: np.ndarray, spec: ModelSpec, method: str, condition: str) -> tuple[np.ndarray | None, pd.DataFrame]:
    """Run bounded nonlinear least squares from many starting points."""
    if least_squares is None:
        return None, pd.DataFrame([{
            "condition": condition,
            "method": method,
            "success": False,
            "message": "scipy is not installed in the active Python environment; install scipy to run bounded multi-start nonlinear least-squares fitting.",
        }])
    freq = np.asarray(freq, float)
    z = np.asarray(z, complex)
    mask = np.isfinite(freq) & np.isfinite(z.real) & np.isfinite(z.imag) & (freq > 0)
    freq = freq[mask]
    z = z[mask]
    order = np.argsort(freq)
    freq = freq[order]
    z = z[order]
    if len(freq) < max(len(spec.param_names) + 3, 12):
        return None, pd.DataFrame([{"condition": condition, "method": method, "success": False, "message": "too few points"}])
    lower, upper = dynamic_bounds(freq, z, spec)
    starts = initial_points(freq, z, spec)
    rows = []
    best_p = None
    best_cost = np.inf
    for sid, x0 in enumerate(starts):
        try:
            res = least_squares(
                residual_vector,
                x0,
                args=(freq, z, spec),
                bounds=(lower, upper),
                loss="soft_l1",
                f_scale=0.05,
                x_scale="jac",
                max_nfev=20000,
            )
            p = sort_relaxation_branches(res.x, spec)
            cost = float(np.sum(residual_vector(p, freq, z, spec) ** 2))
            row = {"condition": condition, "method": method, "start_id": sid, "success": bool(res.success), "cost": cost, "message": str(res.message)}
            row.update({f"fit_{n}": float(v) for n, v in zip(spec.param_names, p)})
            rows.append(row)
            if res.success and cost < best_cost:
                best_cost = cost
                best_p = p
        except Exception as exc:
            rows.append({"condition": condition, "method": method, "start_id": sid, "success": False, "cost": np.inf, "message": repr(exc)})
    return best_p, pd.DataFrame(rows)
def parameter_error_row(p: np.ndarray | None, condition: str, method: str, spec: ModelSpec) -> dict:
    """APE for fitted parameters against known synthetic truth."""
    row = {"condition": condition, "method": method}
    if p is None:
        for name in spec.param_names:
            row[f"{name}_fit"] = np.nan
            row[f"{name}_true"] = float(spec.true_params[spec.param_names.index(name)])
            row[f"{name}_abs_pct_error"] = np.nan
        row["median_param_abs_pct_error"] = np.nan
        row["max_param_abs_pct_error"] = np.nan
        return row
    p = sort_relaxation_branches(np.asarray(p, float), spec)
    apes = []
    for name, est, truth in zip(spec.param_names, p, spec.true_params):
        ape = 100 * abs(est - truth) / (abs(truth) + 1e-30)
        row[f"{name}_fit"] = float(est)
        row[f"{name}_true"] = float(truth)
        row[f"{name}_abs_pct_error"] = float(ape)
        apes.append(ape)
    row["median_param_abs_pct_error"] = float(np.nanmedian(apes))
    row["max_param_abs_pct_error"] = float(np.nanmax(apes))
    return row
# readdata
fit_model_multistart = fit_multistart
def fitted_model_to_exact_full_grid_error(p: np.ndarray | None, truth_df: pd.DataFrame, spec: ModelSpec) -> float:
    """Evaluate fitted parametric model on full truth grid and compare to truth."""
    if p is None:
        return np.nan
    truth_df = clean_curve_df(truth_df)
    freq_full = truth_df["freq"].to_numpy(float)
    z_true = z_from_df(truth_df)
    z_fit_full = spec.model_func(freq_full, p)
    return complex_rel_l2(z_fit_full, z_true)
def read_truth_sheet(xls: pd.ExcelFile) -> pd.DataFrame:
    if "truth" not in xls.sheet_names:
        raise ValueError("Workbook does not contain a 'truth' sheet.")
    return clean_curve_df(pd.read_excel(xls, sheet_name="truth"))
def load_gpht_curves(workbook: Path) -> tuple[pd.DataFrame, dict[tuple[str, str], pd.DataFrame]]:
    """Read GP-HT dual-side workbook and return method curves.
    Output methods:
    - GPHT_imInput: Re is HT prediction from Im input; Im is GP regression.
    - GPHT_reInput: Re is GP regression after R_inf handling; Im is HT prediction.
    - Direct_input: degraded/raw input curve if the sheet has re/imag columns.
    """
    xls = pd.ExcelFile(workbook)
    truth_df = read_truth_sheet(xls)
    curves: dict[tuple[str, str], pd.DataFrame] = {}
    summary_sheets = {"metrics", "optimization_log", "bht_metrics", "linkk_metrics"}
    for sheet in xls.sheet_names:
        if sheet == "truth" or sheet in summary_sheets:
            continue
        df = standardize_columns(pd.read_excel(xls, sheet_name=sheet))
        if {"freq_pred", "imInput_re_pred", "imInput_imag_reg"}.issubset(df.columns):
            curves[(sheet, "GPHT_imInput")] = pd.DataFrame({
                "freq": df["freq_pred"],
                "re": df["imInput_re_pred"],
                "imag": df["imInput_imag_reg"],
                "sd_re": df.get("imInput_re_sd", np.nan),
                "sd_imag": df.get("imInput_imag_sd", np.nan),
            })
        if {"freq_pred", "reInput_re_reg", "reInput_imag_pred"}.issubset(df.columns):
            curves[(sheet, "GPHT_reInput")] = pd.DataFrame({
                "freq": df["freq_pred"],
                "re": df["reInput_re_reg"],
                "imag": df["reInput_imag_pred"],
                "sd_re": df.get("reInput_re_sd", np.nan),
                "sd_imag": df.get("reInput_imag_sd", np.nan),
            })
        if {"freq", "re", "imag"}.issubset(df.columns):
            curves[(sheet, "Direct_input")] = clean_curve_df(df)
    return truth_df, curves
def load_bht_curves(workbook: Path) -> tuple[pd.DataFrame, dict[tuple[str, str], pd.DataFrame]]:
    """Read BHT workbook and return BHT curve variants."""
    xls = pd.ExcelFile(workbook)
    truth_df = read_truth_sheet(xls)
    curves: dict[tuple[str, str], pd.DataFrame] = {}
    skip = {"bht_metrics", "bht_quality_scores", "optimization_log", "truth"}
    for sheet in xls.sheet_names:
        if sheet in skip or not sheet.startswith("bht_"):
            continue
        df = standardize_columns(pd.read_excel(xls, sheet_name=sheet))
        if not {"freq", "bht_reg_re", "bht_reg_imag", "bht_ht_re", "bht_ht_imag"}.issubset(df.columns):
            continue
        cond = sheet.replace("bht_", "", 1)
        curves[(cond, "BHT_regression")] = pd.DataFrame({"freq": df["freq"], "re": df["bht_reg_re"], "imag": df["bht_reg_imag"]})
        curves[(cond, "BHT_HT")] = pd.DataFrame({"freq": df["freq"], "re": df["bht_ht_re"], "imag": df["bht_ht_imag"]})
        # GP-HT regression settings
        curves[(cond, "BHT_reHT_imReg")] = pd.DataFrame({"freq": df["freq"], "re": df["bht_ht_re"], "imag": df["bht_reg_imag"]})
    return truth_df, curves
def load_linkk_curves(workbook: Path) -> tuple[pd.DataFrame, dict[tuple[str, str], pd.DataFrame]]:
    """Read Lin-KK workbook and return fit curves.
    Lin-KK is mainly an in-band consistency diagnostic.  It normally does not
    provide a probabilistic full-band extrapolation; full-grid curve metrics are
    therefore only filled if the stored frequency grid covers the truth range.
    """
    xls = pd.ExcelFile(workbook)
    truth_df = read_truth_sheet(xls)
    curves: dict[tuple[str, str], pd.DataFrame] = {}
    skip = {"linkk_metrics", "linkk_quality_scores", "optimization_log", "truth"}
    for sheet in xls.sheet_names:
        if sheet in skip or not sheet.startswith("linkk_"):
            continue
        df = standardize_columns(pd.read_excel(xls, sheet_name=sheet))
        if {"freq", "linkk_fit_re", "linkk_fit_imag"}.issubset(df.columns):
            cond = sheet.replace("linkk_", "", 1)
            curves[(cond, "LinKK_fit")] = pd.DataFrame({"freq": df["freq"], "re": df["linkk_fit_re"], "imag": df["linkk_fit_imag"]})
        if {"freq", "input_re", "input_imag"}.issubset(df.columns):
            cond = sheet.replace("linkk_", "", 1)
            curves[(cond, "Direct_input")] = pd.DataFrame({"freq": df["freq"], "re": df["input_re"], "imag": df["input_imag"]})
    return truth_df, curves
LOADER_BY_FAMILY = {
    "GPHT": load_gpht_curves,
    "BHT": load_bht_curves,
    "LINKK": load_linkk_curves,
}
def run_recovery_pipeline(
    model_key: str,
    family: str,
    workbook_path: str | Path,
    output_dir: str | Path,
    selected_methods: Iterable[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Complete V3 recovery pipeline for one model/method family."""
    spec = MODEL_SPECS[model_key]
    loader = LOADER_BY_FAMILY[family.upper()]
    workbook_path = Path(workbook_path)
    output_dir = ensure_dir(output_dir)
    table_dir = ensure_dir(output_dir / "tables")
    fig_dir = ensure_dir(output_dir / "figures")
    truth_df, curves = loader(workbook_path)
    if selected_methods is not None:
        selected_methods = set(selected_methods)
        curves = {k: v for k, v in curves.items() if k[1] in selected_methods}
    curve_rows = []
    param_rows = []
    fit_log_frames = []
    for (condition, method), cdf in curves.items():
        cdf = clean_curve_df(cdf)
        curve_rows.append(curve_metric_row(truth_df, cdf, condition, method))
        freq = cdf["freq"].to_numpy(float)
        z = z_from_df(cdf)
        best_p, log_df = fit_multistart(freq, z, spec, method=method, condition=condition)
        log_df["model"] = spec.name
        fit_log_frames.append(log_df)
        row = parameter_error_row(best_p, condition, method, spec)
        row["fitted_model_to_exact_full_grid_error"] = fitted_model_to_exact_full_grid_error(best_p, truth_df, spec)
        row["model_note"] = spec.notes
        param_rows.append(row)
    curve_metrics = pd.DataFrame(curve_rows)
    parameter_recovery = pd.DataFrame(param_rows)
    fit_log = pd.concat(fit_log_frames, ignore_index=True) if fit_log_frames else pd.DataFrame()
    out_xlsx = table_dir / f"{spec.name}_{family}_recoveryV3.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        truth_df.to_excel(writer, sheet_name="truth", index=False)
        curve_metrics.to_excel(writer, sheet_name="curve_metrics", index=False)
        parameter_recovery.to_excel(writer, sheet_name="parameter_recovery", index=False)
        fit_log.to_excel(writer, sheet_name="multistart_fit_log", index=False)
        for (condition, method), cdf in curves.items():
            cdf.to_excel(writer, sheet_name=safe_sheet_name(f"{condition}_{method}"), index=False)
    curve_metrics.to_csv(table_dir / f"{spec.name}_{family}_curve_metricsV3.csv", index=False)
    parameter_recovery.to_csv(table_dir / f"{spec.name}_{family}_parameter_recoveryV3.csv", index=False)
    fit_log.to_csv(table_dir / f"{spec.name}_{family}_multistart_logV3.csv", index=False)
    make_condition_plots(truth_df, curves, parameter_recovery, spec, fig_dir, family)
    make_parameter_ape_plot(parameter_recovery, spec, fig_dir, family)
    make_curve_metric_plot(curve_metrics, fig_dir, family)
    return {
        "truth": truth_df,
        "curve_metrics": curve_metrics,
        "parameter_recovery": parameter_recovery,
        "fit_log": fit_log,
    }
# Plotting settings
def make_condition_plots(
    truth_df: pd.DataFrame,
    curves: dict[tuple[str, str], pd.DataFrame],
    parameter_recovery: pd.DataFrame,
    spec: ModelSpec,
    fig_dir: Path,
    family: str,
) -> None:
    """Save one 3-panel figure per condition.
    Panel 1: Nyquist, x=Z' / Ohm, y=-Z'' / Ohm, equal aspect.
    Panel 2: Re(Z) vs frequency, x=f / Hz log-scale, y=Z' / Ohm.
    Panel 3: -Im(Z) vs frequency, x=f / Hz log-scale, y=-Z'' / Ohm.
    """
    if plt is None:
        (fig_dir / "_figures_not_generated.txt").write_text(
            "matplotlib is not installed in the active Python environment; "
            "all V3 tables were generated, but PNG figures were skipped. "
            "Install matplotlib or run the notebook in an environment that already has it to generate figures.\n",
            encoding="utf-8",
        )
        return
    truth_df = clean_curve_df(truth_df)
    f_truth = truth_df["freq"].to_numpy(float)
    z_truth = z_from_df(truth_df)
    conditions = sorted({cond for cond, _ in curves.keys()})
    colors = {
        "Direct_input": "0.55",
        "GPHT_imInput": "tab:blue",
        "GPHT_reInput": "tab:orange",
        "BHT_reHT_imReg": "tab:purple",
        "BHT_regression": "tab:green",
        "BHT_HT": "tab:red",
        "LinKK_fit": "tab:brown",
    }
    for cond in conditions:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
        ax_nyq, ax_re, ax_im = axes
        ax_nyq.plot(z_truth.real, -z_truth.imag, color="black", lw=2.5, label="exact truth")
        ax_re.semilogx(f_truth, z_truth.real, color="black", lw=2.5, label="exact truth")
        ax_im.semilogx(f_truth, -z_truth.imag, color="black", lw=2.5, label="exact truth")
        for (c, method), cdf in curves.items():
            if c != cond:
                continue
            cdf = clean_curve_df(cdf)
            f = cdf["freq"].to_numpy(float)
            z = z_from_df(cdf)
            style = "o" if method == "Direct_input" else "-"
            lw = 1.3 if method == "Direct_input" else 2.0
            ms = 3 if method == "Direct_input" else 0
            color = colors.get(method, None)
            ax_nyq.plot(z.real, -z.imag, style, lw=lw, ms=ms, color=color, alpha=0.75, label=method)
            ax_re.semilogx(f, z.real, style, lw=lw, ms=ms, color=color, alpha=0.75, label=method)
            ax_im.semilogx(f, -z.imag, style, lw=lw, ms=ms, color=color, alpha=0.75, label=method)
            # Error-metric calculation
            prow = parameter_recovery[(parameter_recovery["condition"] == cond) & (parameter_recovery["method"] == method)]
            if not prow.empty:
                p = np.array([prow.iloc[0].get(f"fit_{n}", np.nan) for n in spec.param_names], float)
                if np.all(np.isfinite(p)):
                    z_fit_full = spec.model_func(f_truth, p)
                    ax_nyq.plot(z_fit_full.real, -z_fit_full.imag, "--", lw=1.4, color=color, alpha=0.75, label=f"{method} param fit")
                    ax_re.semilogx(f_truth, z_fit_full.real, "--", lw=1.4, color=color, alpha=0.75)
                    ax_im.semilogx(f_truth, -z_fit_full.imag, "--", lw=1.4, color=color, alpha=0.75)
        ax_nyq.set_aspect("equal", adjustable="box")
        ax_nyq.set_xlabel("Z' / Ohm")
        ax_nyq.set_ylabel("-Z'' / Ohm")
        ax_nyq.set_title("Nyquist")
        ax_re.set_xlabel("Frequency / Hz")
        ax_re.set_ylabel("Z' / Ohm")
        ax_re.set_title("Real part")
        ax_im.set_xlabel("Frequency / Hz")
        ax_im.set_ylabel("-Z'' / Ohm")
        ax_im.set_title("Imaginary part")
        handles, labels = ax_re.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
        fig.suptitle(f"{spec.name} | {family} | {cond}", y=1.03)
        fig.savefig(fig_dir / f"{spec.name}_{family}_{safe_sheet_name(cond)}_curve_fitV3.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
def make_parameter_ape_plot(parameter_recovery: pd.DataFrame, spec: ModelSpec, fig_dir: Path, family: str) -> None:
    """Grouped bar plot: x=parameter, y=absolute percentage error (%)."""
    if plt is None:
        (fig_dir / "_figures_not_generated.txt").write_text(
            "matplotlib is not installed in the active Python environment; "
            "all V3 tables were generated, but PNG figures were skipped. "
            "Install matplotlib or run the notebook in an environment that already has it to generate figures.\n",
            encoding="utf-8",
        )
        return
    if parameter_recovery.empty:
        return
    cols = [f"{n}_abs_pct_error" for n in spec.param_names if f"{n}_abs_pct_error" in parameter_recovery.columns]
    if not cols:
        return
    long = parameter_recovery.melt(id_vars=["condition", "method"], value_vars=cols, var_name="parameter", value_name="APE_percent")
    long["parameter"] = long["parameter"].str.replace("_abs_pct_error", "", regex=False)
    med = long.groupby(["method", "parameter"], as_index=False)["APE_percent"].median()
    methods = list(med["method"].drop_duplicates())
    x = np.arange(len(spec.param_names))
    width = 0.8 / max(len(methods), 1)
    fig, ax = plt.subplots(figsize=(12, 5))
    for k, method in enumerate(methods):
        vals = []
        for name in spec.param_names:
            s = med[(med["method"] == method) & (med["parameter"] == name)]["APE_percent"]
            vals.append(float(s.iloc[0]) if len(s) else np.nan)
        ax.bar(x + (k - (len(methods) - 1) / 2) * width, vals, width=width, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(spec.param_names, rotation=30, ha="right")
    ax.set_ylabel("Median parameter APE / %")
    ax.set_title(f"{spec.name} {family}: parameter recovery")
    ax.legend(frameon=False)
    ax.set_yscale("log")
    ax.grid(True, axis="y", which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{spec.name}_{family}_parameter_APE_barV3.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
def make_curve_metric_plot(curve_metrics: pd.DataFrame, fig_dir: Path, family: str) -> None:
    """Scatter/bar summary for curve metrics."""
    if plt is None:
        (fig_dir / "_figures_not_generated.txt").write_text(
            "matplotlib is not installed in the active Python environment; "
            "all V3 tables were generated, but PNG figures were skipped. "
            "Install matplotlib or run the notebook in an environment that already has it to generate figures.\n",
            encoding="utf-8",
        )
        return
    if curve_metrics.empty or "available_grid_complex_rel_l2" not in curve_metrics:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    methods = list(curve_metrics["method"].drop_duplicates())
    positions = np.arange(len(methods))
    vals = [curve_metrics.loc[curve_metrics["method"] == m, "available_grid_complex_rel_l2"].median() for m in methods]
    ax.bar(positions, vals, color="tab:blue", alpha=0.75)
    ax.set_xticks(positions)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel("Median available-grid complex relative L2")
    ax.set_title(f"{family}: curve recovery metric summary")
    ax.set_yscale("log")
    ax.grid(True, axis="y", which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{family}_curve_metric_summaryV3.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
