from __future__ import annotations

import gc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


START_DATE = pd.Timestamp("2021-01-01")
END_DATE = pd.Timestamp("2024-05-30")
HORIZONS = [5, 10, 20, 60]
MIN_CROSS_SECTION = 30


def format_ticker(series: pd.Series) -> pd.Series:
    return series.astype(str).str.zfill(6)


def parse_dates(series: pd.Series) -> pd.Series:
    text = series.astype(str)
    if text.str.contains("-").any():
        return pd.to_datetime(text, errors="coerce")
    return pd.to_datetime(text, format="%Y%m%d", errors="coerce")


def load_price_panel(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["ticker", "date", "twap"])
    df["ticker"] = format_ticker(df["ticker"])
    df["date"] = parse_dates(df["date"])
    df = df.dropna(subset=["date"])
    df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)]
    panel = (
        df.pivot(index="date", columns="ticker", values="twap")
        .sort_index()
        .sort_index(axis=1)
    )
    return panel


def load_daily_panel(
    path: Path,
    value_col: str,
    trade_dates: pd.DatetimeIndex,
    all_tickers: pd.Index,
    date_col: str = "date",
) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["ticker", date_col, value_col])
    df["ticker"] = format_ticker(df["ticker"])
    df["date"] = parse_dates(df[date_col])
    df = df.dropna(subset=["date"])
    df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)]
    panel = df.pivot(index="date", columns="ticker", values=value_col).sort_index()
    return panel.reindex(index=trade_dates, columns=all_tickers)


def load_low_freq_panel(
    path: Path,
    value_col: str,
    trade_dates: pd.DatetimeIndex,
    all_tickers: pd.Index,
    date_col: str = "date",
) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["ticker", date_col, value_col])
    df["ticker"] = format_ticker(df["ticker"])
    df["date"] = parse_dates(df[date_col])
    df = df.dropna(subset=["date"])
    df = df[df["date"] <= END_DATE]
    panel = df.pivot(index="date", columns="ticker", values=value_col).sort_index()
    union_index = panel.index.union(trade_dates)
    panel = panel.reindex(union_index).sort_index().ffill()
    return panel.reindex(index=trade_dates, columns=all_tickers)


def load_bvalue_panels(
    path: Path,
    trade_dates: pd.DatetimeIndex,
    all_tickers: pd.Index,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path, usecols=["ticker", "date", "TShEquity"])
    df["ticker"] = format_ticker(df["ticker"])
    df["date"] = parse_dates(df["date"])
    df = df.dropna(subset=["date"]).sort_values(["ticker", "date"])
    df = df[df["date"] <= END_DATE]
    df["B_lag"] = df.groupby("ticker")["TShEquity"].shift(1)

    b_panel = df.pivot(index="date", columns="ticker", values="TShEquity").sort_index()
    b_lag_panel = df.pivot(index="date", columns="ticker", values="B_lag").sort_index()

    union_index = b_panel.index.union(trade_dates)
    b_panel = b_panel.reindex(union_index).sort_index().ffill()
    b_lag_panel = b_lag_panel.reindex(union_index).sort_index().ffill()

    b_panel = b_panel.reindex(index=trade_dates, columns=all_tickers)
    b_lag_panel = b_lag_panel.reindex(index=trade_dates, columns=all_tickers)
    return b_panel, b_lag_panel


def winsorize(values: np.ndarray, lower: float = 0.01, upper: float = 0.99) -> np.ndarray:
    low, high = np.nanquantile(values, [lower, upper])
    return np.clip(values, low, high)


def neutralize_con_np(
    con_np: pd.DataFrame,
    market_value: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    y_arr = con_np.to_numpy(dtype=float)
    mv_arr = market_value.to_numpy(dtype=float)
    resid_arr = np.full_like(y_arr, np.nan)
    sample_counts = np.zeros(len(con_np.index), dtype=int)

    for i in range(len(con_np.index)):
        mv = mv_arr[i]
        y = y_arr[i]
        mask = np.isfinite(mv) & (mv > 0) & np.isfinite(y)
        sample_counts[i] = int(mask.sum())
        if sample_counts[i] < MIN_CROSS_SECTION:
            continue
        xv = np.log(mv[mask])
        yv = y[mask]
        X = np.column_stack([np.ones(mask.sum()), xv])
        beta, _, _, _ = np.linalg.lstsq(X, yv, rcond=None)
        resid_arr[i, mask] = yv - X @ beta

        if i % 100 == 0:
            print(f"Neutralization progress: {i + 1}/{len(con_np.index)}")

    resid_df = pd.DataFrame(resid_arr, index=con_np.index, columns=con_np.columns)
    count_series = pd.Series(sample_counts, index=con_np.index, name="neutralization_n")
    return resid_df, count_series


def build_misv_factor(
    market_value: pd.DataFrame,
    b_value: pd.DataFrame,
    b_lag: pd.DataFrame,
    pb: pd.DataFrame,
    ni_neutral: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    mv_arr = market_value.to_numpy(dtype=float)
    b_arr = b_value.to_numpy(dtype=float)
    b_lag_arr = b_lag.to_numpy(dtype=float)
    pb_arr = pb.to_numpy(dtype=float)
    ni_arr = ni_neutral.to_numpy(dtype=float)
    misv_arr = np.full_like(mv_arr, np.nan)
    sample_counts = np.zeros(len(market_value.index), dtype=int)

    for i in range(len(market_value.index)):
        with np.errstate(divide="ignore", invalid="ignore"):
            y = mv_arr[i] / b_arr[i] - 1.0
            x1 = ni_arr[i] / b_arr[i]
            x2 = b_lag_arr[i] / b_arr[i]

        mask = (
            np.isfinite(y)
            & np.isfinite(x1)
            & np.isfinite(x2)
            & np.isfinite(pb_arr[i])
            & (pb_arr[i] != 0)
            & np.isfinite(b_arr[i])
            & (b_arr[i] != 0)
        )
        sample_counts[i] = int(mask.sum())
        if sample_counts[i] < MIN_CROSS_SECTION:
            continue

        y_w = winsorize(y[mask])
        x1_w = winsorize(x1[mask])
        x2_w = winsorize(x2[mask])
        X = np.column_stack([x1_w, x2_w])
        beta, _, _, _ = np.linalg.lstsq(X, y_w, rcond=None)
        y_hat = X @ beta
        misv_arr[i, mask] = (y_hat + 1.0) / pb_arr[i, mask] - 1.0

        if i % 100 == 0:
            print(f"Factor regression progress: {i + 1}/{len(market_value.index)}")

    factor_df = pd.DataFrame(misv_arr, index=market_value.index, columns=market_value.columns)
    count_series = pd.Series(sample_counts, index=market_value.index, name="factor_regression_n")
    return factor_df, count_series


def compute_ic_series(
    factor_rank: pd.DataFrame,
    twap: pd.DataFrame,
    weights: pd.DataFrame,
    horizon: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    future_ret = twap.shift(-horizon) / twap - 1.0
    x_arr = factor_rank.to_numpy(dtype=float)
    y_arr = future_ret.to_numpy(dtype=float)
    w_arr = weights.to_numpy(dtype=float)

    ic_values: list[float] = []
    wic_values: list[float] = []
    valid_dates: list[pd.Timestamp] = []
    sample_counts = np.zeros(len(factor_rank.index), dtype=int)

    for i, date in enumerate(factor_rank.index):
        x = x_arr[i]
        y = y_arr[i]
        w = w_arr[i]
        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
        sample_counts[i] = int(mask.sum())
        if sample_counts[i] < MIN_CROSS_SECTION:
            continue

        xv = x[mask]
        yv = y[mask]
        wv = w[mask]
        weighted_mean_x = np.average(xv, weights=wv)
        weighted_mean_y = np.average(yv, weights=wv)

        weighted_cov = np.nansum(wv * (xv - weighted_mean_x) * (yv - weighted_mean_y))
        weighted_std_x = np.sqrt(np.nansum(wv * (xv - weighted_mean_x) ** 2))
        weighted_std_y = np.sqrt(np.nansum(wv * (yv - weighted_mean_y) ** 2))
        if weighted_std_x == 0 or weighted_std_y == 0:
            continue
        weighted_ic = weighted_cov / (weighted_std_x * weighted_std_y)

        ic_cov = np.average((xv - weighted_mean_x) * (yv - weighted_mean_y))
        ic_std_x = np.sqrt(np.average((xv - weighted_mean_x) ** 2))
        ic_std_y = np.sqrt(np.average((yv - weighted_mean_y) ** 2))
        if ic_std_x == 0 or ic_std_y == 0:
            continue
        ic = ic_cov / (ic_std_x * ic_std_y)

        valid_dates.append(date)
        ic_values.append(ic)
        wic_values.append(weighted_ic)

    ic_series = pd.Series(ic_values, index=pd.DatetimeIndex(valid_dates), name=f"IC_{horizon}")
    wic_series = pd.Series(wic_values, index=pd.DatetimeIndex(valid_dates), name=f"WIC_{horizon}")
    count_series = pd.Series(sample_counts, index=factor_rank.index, name=f"IC_n_{horizon}")
    return ic_series, wic_series, count_series


def save_diagnostics(
    output_dir: Path,
    diagnostics: pd.DataFrame,
    ic_results: dict[int, tuple[pd.Series, pd.Series, pd.Series]],
) -> None:
    diagnostics.to_csv(output_dir / "daily_diagnostics.csv", encoding="utf-8-sig")

    ic_frames = []
    for horizon, (ic, wic, counts) in ic_results.items():
        ic_frames.append(
            pd.concat(
                [
                    ic.rename(f"IC_{horizon}"),
                    wic.rename(f"WIC_{horizon}"),
                    counts.rename(f"IC_sample_n_{horizon}"),
                ],
                axis=1,
                sort=False,
            )
        )

    ic_panel = pd.concat(ic_frames, axis=1, sort=False).sort_index()
    ic_panel.to_csv(output_dir / "daily_ic_series.csv", encoding="utf-8-sig")


def build_summary(ic_results: dict[int, tuple[pd.Series, pd.Series]]) -> pd.DataFrame:
    rows = []
    for horizon, (ic, wic) in ic_results.items():
        ic_mean = ic.mean()
        ic_std = ic.std()
        wic_mean = wic.mean()
        wic_std = wic.std()
        scale = np.sqrt(252 / horizon)

        rows.append(
            {
                "horizon": horizon,
                "IC均值": ic_mean,
                "IC标准差": ic_std,
                "ICIR": np.nan if pd.isna(ic_std) or ic_std == 0 else ic_mean / ic_std * scale,
                "WIC均值": wic_mean,
                "WICIR": np.nan if pd.isna(wic_std) or wic_std == 0 else wic_mean / wic_std * scale,
                "IC>0占比": (ic > 0).mean(),
            }
        )

    summary = pd.DataFrame(rows).set_index("horizon")
    return summary


def _set_sparse_date_ticks(ax: plt.Axes, dates: pd.DatetimeIndex) -> None:
    if len(dates) == 0:
        return
    if len(dates) <= 6:
        positions = np.arange(len(dates))
    else:
        positions = np.linspace(0, len(dates) - 1, 6, dtype=int)
        positions = np.unique(positions)
    ax.set_xticks(positions)
    ax.set_xticklabels([dates[pos].strftime("%Y-%m") for pos in positions], rotation=0)


def plot_ic_analysis(
    ic_results: dict[int, tuple[pd.Series, pd.Series]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(len(HORIZONS), 2, figsize=(16, 16), constrained_layout=True)

    for row, horizon in enumerate(HORIZONS):
        ic, wic = ic_results[horizon]
        ic = ic.sort_index()
        wic = wic.sort_index()
        dates = ic.index
        positions = np.arange(len(dates))

        bar_ax = axes[row, 0]
        colors = np.where(ic.values >= 0, "#d65f5f", "#4c72b0")
        bar_ax.bar(positions, ic.values, color=colors, width=0.8)
        bar_ax.axhline(ic.mean(), color="#333333", linestyle="--", linewidth=1.2)
        bar_ax.axhline(0, color="#999999", linewidth=0.8)
        bar_ax.set_title(f"{horizon}D Daily IC")
        bar_ax.set_ylabel("IC")
        _set_sparse_date_ticks(bar_ax, dates)

        line_ax = axes[row, 1]
        combined = pd.concat([ic.rename("IC"), wic.rename("WIC")], axis=1).fillna(0.0)
        line_dates = combined.index
        line_positions = np.arange(len(line_dates))
        line_ax.plot(line_positions, combined["IC"].cumsum(), label="Cum IC", color="#4c72b0")
        line_ax.plot(line_positions, combined["WIC"].cumsum(), label="Cum WIC", color="#55a868")
        line_ax.axhline(0, color="#999999", linewidth=0.8)
        line_ax.set_title(f"{horizon}D Cumulative IC vs WIC")
        line_ax.set_ylabel("Cumulative Value")
        _set_sparse_date_ticks(line_ax, line_dates)
        line_ax.legend(loc="best")

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    print(
        "Assumption: low-frequency dates in bValue/con_np_rolling are observable dates; "
        "forward-fill starts on each source date and never before."
    )

    print("Loading price panel...")
    twap = load_price_panel(data_dir / "twap09310940price.csv")
    trade_dates = twap.index
    all_tickers = twap.columns
    print(f"Trade dates: {len(trade_dates)}, tickers: {len(all_tickers)}")

    print("Loading daily panels...")
    market_value = load_daily_panel(
        data_dir / "marketValue.csv",
        value_col="marketValue",
        trade_dates=trade_dates,
        all_tickers=all_tickers,
    )
    pb = load_daily_panel(
        data_dir / "PB.csv",
        value_col="PB",
        trade_dates=trade_dates,
        all_tickers=all_tickers,
    )
    weights = load_daily_panel(
        data_dir / "negMarketValue.csv",
        value_col="negMarketValue",
        trade_dates=trade_dates,
        all_tickers=all_tickers,
        date_col="tradeDate",
    )

    print("Loading low-frequency panels...")
    b_value, b_lag = load_bvalue_panels(
        data_dir / "bValue.csv",
        trade_dates=trade_dates,
        all_tickers=all_tickers,
    )
    con_np = load_low_freq_panel(
        data_dir / "con_np_rolling.csv",
        value_col="con_np_roll",
        trade_dates=trade_dates,
        all_tickers=all_tickers,
    )

    gc.collect()

    print("Neutralizing consensus net profit by market value...")
    ni_neutral, neutral_counts = neutralize_con_np(con_np, market_value)
    del con_np
    gc.collect()

    print("Running daily cross-sectional regression for MISV_FY1...")
    misv_fy1, factor_counts = build_misv_factor(market_value, b_value, b_lag, pb, ni_neutral)
    misv_fy1.to_csv(base_dir / "MISV_FY1.csv", encoding="utf-8-sig")
    misv_fy1_rank = misv_fy1.rank(axis=1, pct=True)
    misv_fy1_rank.to_csv(base_dir / "MISV_FY1_rank.csv", encoding="utf-8-sig")
    print("Saved MISV_FY1.csv")
    print("Saved MISV_FY1_rank.csv")

    print("Computing IC and WIC series...")
    ic_results: dict[int, tuple[pd.Series, pd.Series, pd.Series]] = {}
    for horizon in HORIZONS:
        ic_results[horizon] = compute_ic_series(misv_fy1_rank, twap, weights, horizon)
        print(
            f"Horizon {horizon}: valid dates={len(ic_results[horizon][0])}, "
            f"IC mean={ic_results[horizon][0].mean():.6f}, "
            f"WIC mean={ic_results[horizon][1].mean():.6f}"
        )

    summary = build_summary({k: (v[0], v[1]) for k, v in ic_results.items()})
    summary.to_csv(base_dir / "ic_summary.csv", encoding="utf-8-sig")
    print("\nIC summary:")
    print(summary.round(6).to_string())
    print("Saved ic_summary.csv")

    diagnostics = pd.DataFrame(
        {
            "neutralization_n": neutral_counts,
            "factor_regression_n": factor_counts,
            "factor_valid_n": misv_fy1.notna().sum(axis=1),
        }
    )
    for horizon in HORIZONS:
        diagnostics[f"IC_sample_n_{horizon}"] = ic_results[horizon][2]
    save_diagnostics(base_dir, diagnostics, ic_results)
    print("Saved daily_diagnostics.csv")
    print("Saved daily_ic_series.csv")

    plot_ic_analysis({k: (v[0], v[1]) for k, v in ic_results.items()}, base_dir / "ic_analysis.png")
    print("Saved ic_analysis.png")


if __name__ == "__main__":
    main()
