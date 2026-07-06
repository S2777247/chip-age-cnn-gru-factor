import os
import numpy as np
import pandas as pd


# ================= 1. 参数设置 =================

INPUT_PATH = "data/csi1000_daily_data_20200101_20260423_n300.csv"
OUTPUT_PATH = "data/csi1000_chip_age_factor_n300.csv"

PRICE_BINS_ABS = 300
WARMUP_DAYS = 250


# ================= 2. 构造 VWAP 中心三角分布 =================

def triangular_distribution(price_grid, low, high, vwap):
    """
    用当日 low、high、vwap 构造一个三角形分布。
    作用：近似表示当日新增成交筹码在价格区间上的分布。
    """
    weights = np.zeros_like(price_grid, dtype=float)

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

    if weights.sum() <= 0:
        idx = np.argmin(np.abs(price_grid - vwap))
        weights[idx] = 1.0
    else:
        weights = weights / weights.sum()

    return weights


# ================= 3. 计算单只股票的筹码龄因子 =================

def build_one_stock_chip_age(df_one):
    """
    输入：单只股票的日行情数据
    输出：该股票每日筹码龄因子
    """

    df_one = df_one.sort_values("trade_date").reset_index(drop=True).copy()

    if len(df_one) < WARMUP_DAYS + 30:
        return pd.DataFrame()

    # ---------- 3.1 前复权处理 ----------
    latest_adj = df_one["adj_factor"].iloc[-1]
    if pd.isna(latest_adj) or latest_adj == 0:
        return pd.DataFrame()

    adj_ratio = df_one["adj_factor"] / latest_adj

    df_one["open_qfq"] = df_one["open"] * adj_ratio
    df_one["high_qfq"] = df_one["high"] * adj_ratio
    df_one["low_qfq"] = df_one["low"] * adj_ratio
    df_one["close_qfq"] = df_one["close"] * adj_ratio

    # Tushare: amount 通常是千元，vol 是手，一手是100股
    df_one["vwap_raw"] = df_one["amount"] * 1000 / (df_one["vol"] * 100)
    df_one["vwap_qfq"] = df_one["vwap_raw"] * adj_ratio

    df_one = df_one.replace([np.inf, -np.inf], np.nan)

    df_one = df_one.dropna(subset=[
        "high_qfq",
        "low_qfq",
        "close_qfq",
        "vwap_qfq",
        "turnover"
    ]).reset_index(drop=True)

    if len(df_one) < WARMUP_DAYS + 30:
        return pd.DataFrame()

    # ---------- 3.2 建立价格网格 ----------
    price_min = df_one["low_qfq"].min() * 0.8
    price_max = df_one["high_qfq"].max() * 1.2

    if price_max <= price_min:
        return pd.DataFrame()

    price_grid = np.linspace(price_min, price_max, PRICE_BINS_ABS)

    n = len(df_one)

    # 行：筹码年龄；列：价格网格
    # age_mat[0] 表示 1日龄筹码
    # age_mat[1] 表示 2日龄筹码
    # age_mat[2] 表示 3日龄筹码
    age_mat = np.zeros((n + 1, PRICE_BINS_ABS), dtype=float)

    records = []
    initialized = False

    # ---------- 3.3 逐日递推筹码龄 ----------
    for i, row in df_one.iterrows():
        low = row["low_qfq"]
        high = row["high_qfq"]
        vwap = row["vwap_qfq"]
        close_price = row["close_qfq"]

        tau = row["turnover"]

        if pd.isna(tau):
            continue

        tau = max(0.0, min(float(tau), 0.95))

        daily_new = triangular_distribution(price_grid, low, high, vwap)

        if daily_new.sum() <= 0:
            continue

        if not initialized:
            # 第一天没有历史筹码，只能用当日成交分布初始化
            age_mat[0, :] = daily_new
            initialized = True
        else:
            old = age_mat.copy()
            age_mat[:, :] = 0.0

            # 历史筹码留存，并年龄 +1
            age_mat[1:, :] = old[:-1, :] * (1 - tau)

            # 当日新增筹码进入 1 日龄
            age_mat[0, :] = tau * daily_new

            # 归一化，保证筹码总质量为 1
            total = age_mat.sum()
            if total > 0:
                age_mat /= total

        total_chip = age_mat.sum()
        if total_chip <= 0:
            continue

        # ---------- 3.4 四类筹码龄占比 ----------
        age_1_2 = age_mat[0:2, :].sum()
        age_3_10 = age_mat[2:10, :].sum()
        age_11_100 = age_mat[10:100, :].sum()
        age_101_plus = age_mat[100:, :].sum()

        # ---------- 3.5 额外筹码结构特征 ----------
        total_dist = age_mat.sum(axis=0)

        # 盈利筹码占比：成本价 <= 当前收盘价
        profit_ratio = total_dist[price_grid <= close_price].sum()

        # 平均持仓成本 / 当前收盘价
        avg_cost = (price_grid * total_dist).sum()
        avg_cost_close_ratio = avg_cost / close_price if close_price > 0 else np.nan

        # 筹码峰位置 / 当前收盘价
        peak_idx = np.argmax(total_dist)
        peak_price = price_grid[peak_idx]
        peak_price_close_ratio = peak_price / close_price if close_price > 0 else np.nan

        # 平均筹码龄
        age_weights = age_mat.sum(axis=1)
        age_index = np.arange(1, len(age_weights) + 1)
        avg_chip_age = (age_index * age_weights).sum()

        records.append({
            "ts_code": row["ts_code"],
            "trade_date": row["trade_date"],
            "close_qfq": close_price,

            "age_1_2": age_1_2,
            "age_3_10": age_3_10,
            "age_11_100": age_11_100,
            "age_101_plus": age_101_plus,

            "profit_ratio": profit_ratio,
            "avg_cost_close_ratio": avg_cost_close_ratio,
            "peak_price_close_ratio": peak_price_close_ratio,
            "avg_chip_age": avg_chip_age,

            "turnover": tau
        })

    factor_df = pd.DataFrame(records)

    if factor_df.empty:
        return pd.DataFrame()

    factor_df = factor_df.sort_values("trade_date").reset_index(drop=True)

    # 去掉预热期，避免一开始长期筹码层失真
    factor_df = factor_df.iloc[WARMUP_DAYS:].reset_index(drop=True)

    if factor_df.empty:
        return pd.DataFrame()

    # ---------- 3.6 计算未来收益 ----------
    factor_df["future_5d_ret"] = factor_df["close_qfq"].shift(-5) / factor_df["close_qfq"] - 1
    factor_df["future_10d_ret"] = factor_df["close_qfq"].shift(-10) / factor_df["close_qfq"] - 1
    factor_df["future_20d_ret"] = factor_df["close_qfq"].shift(-20) / factor_df["close_qfq"] - 1

    factor_df = factor_df.dropna(subset=[
        "future_5d_ret",
        "future_10d_ret",
        "future_20d_ret"
    ]).reset_index(drop=True)

    return factor_df


# ================= 4. 主程序 =================

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"找不到输入文件：{INPUT_PATH}")

    print("正在读取数据：", INPUT_PATH)

    df = pd.read_csv(INPUT_PATH)

    df["trade_date"] = pd.to_datetime(df["trade_date"])

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "vol",
        "amount",
        "turnover",
        "adj_factor"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)

    df = df.dropna(subset=[
        "ts_code",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "vol",
        "amount",
        "turnover",
        "adj_factor"
    ])

    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    stock_list = sorted(df["ts_code"].unique())

    print("输入股票数量：", len(stock_list))
    print("输入总行数：", len(df))

    all_factors = []
    failed_stocks = []

    for i, ts_code in enumerate(stock_list, start=1):
        print(f"[{i}/{len(stock_list)}] 正在构造筹码龄因子：{ts_code}")

        df_one = df[df["ts_code"] == ts_code].copy()

        try:
            factor_one = build_one_stock_chip_age(df_one)

            if factor_one.empty:
                print(f"  跳过：{ts_code}，有效数据不足或计算为空")
                failed_stocks.append(ts_code)
                continue

            all_factors.append(factor_one)
            print(f"  成功：{len(factor_one)} 行")

        except Exception as e:
            print(f"  失败：{ts_code}")
            print(f"  原因：{e}")
            failed_stocks.append(ts_code)

    if len(all_factors) == 0:
        raise ValueError("没有成功构造任何股票的筹码龄因子。")

    factor_df = pd.concat(all_factors, ignore_index=True)

    factor_df = factor_df.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)

    factor_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    failed_path = "data/failed_chip_age_factor_stocks_n300.csv"
    pd.DataFrame({"failed_ts_code": failed_stocks}).to_csv(
        failed_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n========== 筹码龄因子构造完成 ==========")
    print("成功股票数量：", factor_df["ts_code"].nunique())
    print("交易日数量：", factor_df["trade_date"].nunique())
    print("总行数：", len(factor_df))
    print("保存路径：", OUTPUT_PATH)
    print("失败股票数量：", len(failed_stocks))
    print("失败股票文件：", failed_path)

    print("\n前5行：")
    print(factor_df.head())

    print("\n字段列表：")
    print(factor_df.columns.tolist())