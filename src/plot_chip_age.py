import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tushare as ts
from tushare.pro import client as _ts_client

# ========== 1. Tushare 设置 ==========
_ts_client.DataApi._DataApi__http_url = "https://tu.brze.top"

TOKEN = os.getenv("TUSHARE_TOKEN") or input("请输入你的 Tushare Token：").strip()
pro = ts.pro_api(TOKEN)

TS_CODE = "002594.SZ"       # 先用比亚迪跑通
START_DATE = "20200101"
END_DATE = "20260423"

PRICE_BINS_ABS = 300        # 内部绝对价格网格
REL_BINS = 32               # 报告里用 32 个相对价格 bin
WARMUP_DAYS = 250           # 前面一段用于筹码递推预热，不用于正式解释


# ========== 2. 拉取数据 ==========
def get_stock_data(ts_code, start_date, end_date):
    daily = pro.daily(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date
    )

    time.sleep(0.6)

    basic = pro.daily_basic(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
        fields="ts_code,trade_date,turnover_rate,turnover_rate_f"
    )

    time.sleep(0.6)

    adj = pro.adj_factor(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date
    )

    df = daily.merge(basic, on=["ts_code", "trade_date"], how="left")
    df = df.merge(adj, on=["ts_code", "trade_date"], how="left")

    df = df.sort_values("trade_date").reset_index(drop=True)

    numeric_cols = [
        "open", "high", "low", "close", "vol", "amount",
        "turnover_rate", "turnover_rate_f", "adj_factor"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[
        "open", "high", "low", "close", "vol", "amount", "adj_factor"
    ])

    # 复权处理：把历史价格调整到最后一天口径
    latest_adj = df["adj_factor"].iloc[-1]
    adj_ratio = df["adj_factor"] / latest_adj

    for col in ["open", "high", "low", "close"]:
        df[col + "_qfq"] = df[col] * adj_ratio

    # Tushare: amount 通常是千元，vol 是手，一手 100 股
    df["vwap_raw"] = df["amount"] * 1000 / (df["vol"] * 100)
    df["vwap_qfq"] = df["vwap_raw"] * adj_ratio

    # 优先用自由流通换手率，没有就用普通换手率
    df["turnover"] = df["turnover_rate_f"].fillna(df["turnover_rate"]) / 100
    df["turnover"] = df["turnover"].clip(lower=0, upper=0.95)

    df = df.dropna(subset=[
        "high_qfq", "low_qfq", "close_qfq", "vwap_qfq", "turnover"
    ])

    return df


# ========== 3. 构造当日新增筹码：VWAP 中心三角分布 ==========
def triangular_distribution(price_grid, low, high, vwap):
    weights = np.zeros_like(price_grid)

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


# ========== 4. 把绝对价格筹码映射到相对当前收盘价 ==========
def map_to_relative_bins(layer_dict, price_grid, close_price, rel_edges):
    rel_pos = np.log(price_grid / close_price)

    valid = (rel_pos >= rel_edges[0]) & (rel_pos <= rel_edges[-1])
    idx = np.digitize(rel_pos[valid], rel_edges) - 1
    idx = np.clip(idx, 0, len(rel_edges) - 2)

    result = {}

    for name, mass in layer_dict.items():
        arr = np.zeros(len(rel_edges) - 1)
        np.add.at(arr, idx, mass[valid])
        result[name] = arr

    total_mass = sum(v.sum() for v in result.values())
    if total_mass > 0:
        for name in result:
            result[name] = result[name] / total_mass

    return result


# ========== 5. 递推筹码龄结构 ==========
def build_chip_age_structure(df):
    price_min = df["low_qfq"].min() * 0.8
    price_max = df["high_qfq"].max() * 1.2
    price_grid = np.linspace(price_min, price_max, PRICE_BINS_ABS)

    rel_edges = np.linspace(-0.7, 0.7, REL_BINS + 1)
    rel_centers = (rel_edges[:-1] + rel_edges[1:]) / 2

    n = len(df)

    # 行：筹码年龄；列：价格网格
    age_mat = np.zeros((n + 1, PRICE_BINS_ABS))

    ratio_records = []
    latest_layers = None
    latest_date = None

    initialized = False

    for i, row in df.iterrows():
        low = row["low_qfq"]
        high = row["high_qfq"]
        vwap = row["vwap_qfq"]
        close_price = row["close_qfq"]
        tau = row["turnover"]

        daily_new = triangular_distribution(price_grid, low, high, vwap)

        if not initialized:
            # 第一天没有历史筹码，只能用当天分布初始化
            age_mat[0, :] = daily_new
            initialized = True
        else:
            old = age_mat.copy()
            age_mat[:, :] = 0.0

            # 历史筹码留存，并年龄 +1
            age_mat[1:, :] = old[:-1, :] * (1 - tau)

            # 当日新增筹码进入年龄 1
            age_mat[0, :] = tau * daily_new

            # 归一化，保持筹码总和为 1
            total = age_mat.sum()
            if total > 0:
                age_mat /= total

        # 四个筹码龄分层
        layer_dict_abs = {
            "age_1_2": age_mat[0:2, :].sum(axis=0),
            "age_3_10": age_mat[2:10, :].sum(axis=0),
            "age_11_100": age_mat[10:100, :].sum(axis=0),
            "age_101_plus": age_mat[100:, :].sum(axis=0),
        }

        # 每层占比
        ratio_records.append({
            "trade_date": row["trade_date"],
            "age_1_2": layer_dict_abs["age_1_2"].sum(),
            "age_3_10": layer_dict_abs["age_3_10"].sum(),
            "age_11_100": layer_dict_abs["age_11_100"].sum(),
            "age_101_plus": layer_dict_abs["age_101_plus"].sum(),
        })

        # 保存最后一天用于画横截面筹码图
        latest_layers = map_to_relative_bins(
            layer_dict_abs,
            price_grid,
            close_price,
            rel_edges
        )
        latest_date = row["trade_date"]

    ratio_df = pd.DataFrame(ratio_records)
    ratio_df["trade_date"] = pd.to_datetime(ratio_df["trade_date"])

    return latest_layers, latest_date, rel_centers, ratio_df


# ========== 6. 画最后一天筹码龄分布图 ==========
def plot_latest_chip_structure(latest_layers, latest_date, rel_centers, ts_code):
    plt.figure(figsize=(12, 6))

    bottom = np.zeros_like(rel_centers)
    width = rel_centers[1] - rel_centers[0]

    order = ["age_101_plus", "age_11_100", "age_3_10", "age_1_2"]

    for name in order:
        values = latest_layers[name]
        plt.bar(
            rel_centers,
            values,
            width=width,
            bottom=bottom,
            label=name,
            alpha=0.85
        )
        bottom += values

    plt.axvline(0, linestyle="--", linewidth=1)
    plt.title(f"Chip Age Structure of {ts_code} on {latest_date}")
    plt.xlabel("log(cost price / current close)")
    plt.ylabel("chip mass")
    plt.legend()
    plt.tight_layout()

    filename = f"chip_age_structure_{ts_code}_{latest_date}.png"
    plt.savefig(filename, dpi=200)
    plt.show()

    print(f"已保存图片：{filename}")


# ========== 7. 画四类筹码占比随时间变化 ==========
def plot_age_ratio_timeseries(ratio_df, ts_code):
    plot_df = ratio_df.iloc[WARMUP_DAYS:].copy()

    plt.figure(figsize=(12, 6))

    for col in ["age_1_2", "age_3_10", "age_11_100", "age_101_plus"]:
        plt.plot(plot_df["trade_date"], plot_df[col], label=col)

    plt.title(f"Chip Age Ratio Time Series of {ts_code}")
    plt.xlabel("date")
    plt.ylabel("ratio")
    plt.legend()
    plt.tight_layout()

    filename = f"chip_age_ratio_{ts_code}.png"
    plt.savefig(filename, dpi=200)
    plt.show()

    print(f"已保存图片：{filename}")


# ========== 8. 主程序 ==========
if __name__ == "__main__":
    df = get_stock_data(TS_CODE, START_DATE, END_DATE)

    print("数据行数：", len(df))
    print(df.head())

    latest_layers, latest_date, rel_centers, ratio_df = build_chip_age_structure(df)

    plot_latest_chip_structure(latest_layers, latest_date, rel_centers, TS_CODE)
    plot_age_ratio_timeseries(ratio_df, TS_CODE)

    ratio_df.to_csv(f"chip_age_ratio_{TS_CODE}.csv", index=False, encoding="utf-8-sig")
    print(f"已保存数据：chip_age_ratio_{TS_CODE}.csv")
    last_row = ratio_df.iloc[-1]

print("最后一个交易日：", last_row["trade_date"])
print("超短期筹码占比 age_1_2：", last_row["age_1_2"])
print("短期筹码占比 age_3_10：", last_row["age_3_10"])
print("中期筹码占比 age_11_100：", last_row["age_11_100"])
print("长期筹码占比 age_101_plus：", last_row["age_101_plus"])

