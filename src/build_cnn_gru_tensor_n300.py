import os
import numpy as np
import pandas as pd


INPUT_PATH = "data/csi1000_daily_data_20200101_20260423_n300.csv"

TENSOR_PATH = "data/csi1000_chip_tensor_n300.npy"
LABEL_PATH = "data/csi1000_label_10d_n300.npy"
DATE_PATH = "data/csi1000_tensor_dates_n300.csv"
STOCK_PATH = "data/csi1000_tensor_stocks_n300.csv"

PRICE_BINS_ABS = 300
REL_BINS = 32
MAX_AGE = 260
WARMUP_DAYS = 250

AGE_CHANNELS = ["age_1_2", "age_3_10", "age_11_100", "age_101_plus"]


def triangular_distribution(price_grid, low, high, vwap):
    weights = np.zeros_like(price_grid, dtype=np.float32)

    if pd.isna(low) or pd.isna(high) or pd.isna(vwap):
        return weights

    if high <= low:
        idx = np.argmin(np.abs(price_grid - vwap))
        weights[idx] = 1.0
        return weights

    vwap = np.clip(vwap, low + 1e-8, high - 1e-8)

    left_mask = (price_grid >= low) & (price_grid <= vwap)
    right_mask = (price_grid > vwap) & (price_grid <= high)

    weights[left_mask] = (price_grid[left_mask] - low) / (vwap - low)
    weights[right_mask] = (high - price_grid[right_mask]) / (high - vwap)

    total = weights.sum()
    if total > 0:
        weights = weights / total
    else:
        idx = np.argmin(np.abs(price_grid - vwap))
        weights[idx] = 1.0

    return weights.astype(np.float32)


def map_to_relative_bins(layer_dict, price_grid, close_price, rel_edges):
    result = np.zeros((4, REL_BINS), dtype=np.float32)

    if pd.isna(close_price) or close_price <= 0:
        return result

    rel_pos = np.log(price_grid / close_price)

    valid = (rel_pos >= rel_edges[0]) & (rel_pos <= rel_edges[-1])
    bin_idx = np.digitize(rel_pos[valid], rel_edges) - 1
    bin_idx = np.clip(bin_idx, 0, REL_BINS - 1)

    for c, name in enumerate(AGE_CHANNELS):
        mass = layer_dict[name]
        arr = np.zeros(REL_BINS, dtype=np.float32)
        np.add.at(arr, bin_idx, mass[valid])
        result[c] = arr

    total = result.sum()
    if total > 0:
        result = result / total

    result = np.sqrt(result)
    return result.astype(np.float32)


def build_one_stock_tensor(df_one, global_dates):
    df_one = df_one.sort_values("trade_date").reset_index(drop=True).copy()

    if len(df_one) < WARMUP_DAYS + 40:
        return None, None

    latest_adj = df_one["adj_factor"].iloc[-1]
    if pd.isna(latest_adj) or latest_adj == 0:
        return None, None

    adj_ratio = df_one["adj_factor"] / latest_adj

    df_one["open_qfq"] = df_one["open"] * adj_ratio
    df_one["high_qfq"] = df_one["high"] * adj_ratio
    df_one["low_qfq"] = df_one["low"] * adj_ratio
    df_one["close_qfq"] = df_one["close"] * adj_ratio

    df_one["vwap_raw"] = df_one["amount"] * 1000 / (df_one["vol"] * 100)
    df_one["vwap_qfq"] = df_one["vwap_raw"] * adj_ratio

    df_one = df_one.replace([np.inf, -np.inf], np.nan)
    df_one = df_one.dropna(
        subset=["high_qfq", "low_qfq", "close_qfq", "vwap_qfq", "turnover"]
    ).reset_index(drop=True)

    if len(df_one) < WARMUP_DAYS + 40:
        return None, None

    price_min = df_one["low_qfq"].min() * 0.8
    price_max = df_one["high_qfq"].max() * 1.2

    if price_max <= price_min:
        return None, None

    price_grid = np.linspace(price_min, price_max, PRICE_BINS_ABS).astype(np.float32)
    rel_edges = np.linspace(-0.7, 0.7, REL_BINS + 1).astype(np.float32)

    date_to_idx = {d: i for i, d in enumerate(global_dates)}

    tensor_one = np.full((len(global_dates), 4, REL_BINS), np.nan, dtype=np.float32)
    label_one = np.full(len(global_dates), np.nan, dtype=np.float32)

    close_arr = df_one["close_qfq"].values.astype(np.float32)
    future_10d_ret = np.full(len(df_one), np.nan, dtype=np.float32)
    future_10d_ret[:-10] = close_arr[10:] / close_arr[:-10] - 1

    age_mat = np.zeros((MAX_AGE, PRICE_BINS_ABS), dtype=np.float32)
    initialized = False

    for i, row in df_one.iterrows():
        low = row["low_qfq"]
        high = row["high_qfq"]
        vwap = row["vwap_qfq"]
        close_price = row["close_qfq"]

        tau = row["turnover"]
        tau = max(0.0, min(float(tau), 0.95))

        daily_new = triangular_distribution(price_grid, low, high, vwap)

        if daily_new.sum() <= 0:
            continue

        if not initialized:
            age_mat[0, :] = daily_new
            initialized = True
        else:
            old = age_mat.copy()
            age_mat[:, :] = 0.0

            age_mat[1:-1, :] = old[:-2, :] * (1 - tau)
            age_mat[-1, :] = (old[-2, :] + old[-1, :]) * (1 - tau)
            age_mat[0, :] = tau * daily_new

            total = age_mat.sum()
            if total > 0:
                age_mat /= total

        if i < WARMUP_DAYS:
            continue

        layer_dict = {
            "age_1_2": age_mat[0:2, :].sum(axis=0),
            "age_3_10": age_mat[2:10, :].sum(axis=0),
            "age_11_100": age_mat[10:100, :].sum(axis=0),
            "age_101_plus": age_mat[100:, :].sum(axis=0),
        }

        chip_tensor = map_to_relative_bins(layer_dict, price_grid, close_price, rel_edges)

        trade_date = row["trade_date"]

        if trade_date in date_to_idx:
            idx = date_to_idx[trade_date]
            tensor_one[idx] = chip_tensor
            label_one[idx] = future_10d_ret[i]

    return tensor_one, label_one


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    print("读取原始数据：", INPUT_PATH)

    df = pd.read_csv(INPUT_PATH)

    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")

    numeric_cols = [
        "open", "high", "low", "close",
        "vol", "amount", "turnover", "adj_factor"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(
        subset=[
            "ts_code", "trade_date", "open", "high", "low", "close",
            "vol", "amount", "turnover", "adj_factor"
        ]
    )

    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    stocks = sorted(df["ts_code"].unique())
    dates = sorted(df["trade_date"].unique())

    print("股票数量：", len(stocks))
    print("交易日数量：", len(dates))

    all_tensor = np.full((len(dates), len(stocks), 4, REL_BINS), np.nan, dtype=np.float32)
    all_label = np.full((len(dates), len(stocks)), np.nan, dtype=np.float32)

    failed = []

    for stock_idx, ts_code in enumerate(stocks):
        print(f"[{stock_idx + 1}/{len(stocks)}] 构造张量：{ts_code}")

        df_one = df[df["ts_code"] == ts_code].copy()

        try:
            tensor_one, label_one = build_one_stock_tensor(df_one, dates)

            if tensor_one is None:
                failed.append(ts_code)
                print("  跳过：数据不足")
                continue

            all_tensor[:, stock_idx, :, :] = tensor_one
            all_label[:, stock_idx] = label_one

            print("  有效标签数量：", np.isfinite(label_one).sum())

        except Exception as e:
            failed.append(ts_code)
            print("  失败：", e)

    np.save(TENSOR_PATH, all_tensor)
    np.save(LABEL_PATH, all_label)

    pd.DataFrame({"trade_date": dates}).to_csv(DATE_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame({"ts_code": stocks}).to_csv(STOCK_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame({"failed_ts_code": failed}).to_csv(
        "data/failed_tensor_stocks_n300.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n========== CNN-GRU 输入张量构造完成 ==========")
    print("tensor shape:", all_tensor.shape)
    print("label shape:", all_label.shape)
    print("tensor保存：", TENSOR_PATH)
    print("label保存：", LABEL_PATH)
    print("失败股票数量：", len(failed))