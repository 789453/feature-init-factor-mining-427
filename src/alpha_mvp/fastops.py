from __future__ import annotations
import numpy as np
from numba import njit, prange

@njit(cache=True)
def _rank_cs_2d_impl(x):
    n, k = x.shape
    out = np.empty_like(x)
    for i in range(n):
        valid_count = 0
        for j in range(k):
            if np.isfinite(x[i, j]):
                valid_count += 1
        if valid_count == 0:
            for j in range(k):
                out[i, j] = np.nan
            continue
        ranks = np.empty(valid_count, dtype=np.float64)
        vals = np.empty(valid_count, dtype=np.float64)
        idx = 0
        for j in range(k):
            if np.isfinite(x[i, j]):
                vals[idx] = x[i, j]
                idx += 1
        sort_idx = np.argsort(vals)
        for j in range(valid_count):
            ranks[sort_idx[j]] = (j + 1.0) / valid_count
        idx = 0
        for j in range(k):
            if np.isfinite(x[i, j]):
                out[i, j] = ranks[idx]
                idx += 1
            else:
                out[i, j] = np.nan
    return out

@njit(cache=True)
def _rank_cs_1d_impl(x):
    n = len(x)
    out = np.empty(n, dtype=np.float64)
    valid_count = 0
    for i in range(n):
        if np.isfinite(x[i]):
            valid_count += 1
    if valid_count == 0:
        for i in range(n):
            out[i] = np.nan
        return out
    vals = np.empty(valid_count, dtype=np.float64)
    idx = 0
    for i in range(n):
        if np.isfinite(x[i]):
            vals[idx] = x[i]
            idx += 1
    sort_idx = np.argsort(vals)
    ranks = np.empty(valid_count, dtype=np.float64)
    for j in range(valid_count):
        ranks[sort_idx[j]] = (j + 1.0) / valid_count
    idx = 0
    for i in range(n):
        if np.isfinite(x[i]):
            out[i] = ranks[idx]
            idx += 1
        else:
            out[i] = np.nan
    return out

def rank_cs(x):
    if x.ndim == 1:
        return _rank_cs_1d_impl(x)
    return _rank_cs_2d_impl(x)

@njit(cache=True, parallel=True)
def _daily_corr_impl(x, y, rank_flag):
    n = x.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in prange(n):
        xi = x[i]
        yi = y[i]
        valid = 0
        for j in range(len(xi)):
            if np.isfinite(xi[j]) and np.isfinite(yi[j]):
                valid += 1
        if valid < 10:
            out[i] = np.nan
            continue
        fx = np.empty(valid, dtype=np.float64)
        fy = np.empty(valid, dtype=np.float64)
        cnt = 0
        for j in range(len(xi)):
            if np.isfinite(xi[j]) and np.isfinite(yi[j]):
                fx[cnt] = xi[j]
                fy[cnt] = yi[j]
                cnt += 1
        if rank_flag:
            sort_idx_x = np.argsort(fx)
            sort_idx_y = np.argsort(fy)
            rx = np.empty(valid, dtype=np.float64)
            ry = np.empty(valid, dtype=np.float64)
            for j in range(valid):
                rx[sort_idx_x[j]] = (j + 1.0) / valid
                ry[sort_idx_y[j]] = (j + 1.0) / valid
            mx, my = 0.0, 0.0
            for j in range(valid):
                mx += rx[j]
                my += ry[j]
            mx /= valid
            my /= valid
            num = 0.0
            dx, dy = 0.0, 0.0
            for j in range(valid):
                dx += (rx[j] - mx) * (rx[j] - mx)
                dy += (ry[j] - my) * (ry[j] - my)
                num += (rx[j] - mx) * (ry[j] - my)
            denom = np.sqrt(dx * dy)
            out[i] = num / denom if denom > 1e-10 else np.nan
        else:
            mx, my = 0.0, 0.0
            for j in range(valid):
                mx += fx[j]
                my += fy[j]
            mx /= valid
            my /= valid
            num = 0.0
            dx, dy = 0.0, 0.0
            for j in range(valid):
                dx += (fx[j] - mx) * (fx[j] - mx)
                dy += (fy[j] - my) * (fy[j] - my)
                num += (fx[j] - mx) * (fy[j] - my)
            denom = np.sqrt(dx * dy)
            out[i] = num / denom if denom > 1e-10 else np.nan
    return out

def daily_corr(x, y, rank=False):
    return _daily_corr_impl(x, y, rank)

@njit(cache=True, parallel=True)
def _rolling_corr_impl(x, y, w):
    n, k = x.shape
    out = np.full((n, k), np.nan, dtype=np.float64)
    for j in prange(k):
        for i in range(w - 1, n):
            valid = 0
            for t in range(i - w + 1, i + 1):
                if np.isfinite(x[t, j]) and np.isfinite(y[t, j]):
                    valid += 1
            if valid < max(2, w // 2):
                continue
            fx = np.empty(valid, dtype=np.float64)
            fy = np.empty(valid, dtype=np.float64)
            cnt = 0
            for t in range(i - w + 1, i + 1):
                if np.isfinite(x[t, j]) and np.isfinite(y[t, j]):
                    fx[cnt] = x[t, j]
                    fy[cnt] = y[t, j]
                    cnt += 1
            mx, my = 0.0, 0.0
            for ii in range(valid):
                mx += fx[ii]
                my += fy[ii]
            mx /= valid
            my /= valid
            num = 0.0
            dx, dy = 0.0, 0.0
            for ii in range(valid):
                dx += (fx[ii] - mx) * (fx[ii] - mx)
                dy += (fy[ii] - my) * (fy[ii] - my)
                num += (fx[ii] - mx) * (fy[ii] - my)
            denom = np.sqrt(dx * dy)
            out[i, j] = num / denom if denom > 1e-10 else np.nan
    return out

def rolling_corr(x, y, w):
    return _rolling_corr_impl(x, y, w)

def fast_rank_cs(x):
    return rank_cs(x)

def fast_daily_corr(x, y, rank=False):
    return daily_corr(x, y, rank)

def fast_rolling_corr(x, y, w):
    return rolling_corr(x, y, w)