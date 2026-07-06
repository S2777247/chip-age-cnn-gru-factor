import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ================= 1. 参数设置 =================

INPUT_PATH = "data/csi1000_chip_age_factor_n300_tradable.csv"

OUTPUT_DAILY_PATH = "data/csi1000_rankic_daily_n300_tradable.csv"
OUTPUT_SUMMARY_PATH = "data/csi1000_rankic_summary_n300_tradable.csv"
OUTPUT_FIG_PATH = "data/csi1000_rankic_cumsum_future10d_n300_tradable.png"

MIN_STOCKS_PER_DAY = 30

FACTOR_COLS = [
    "age_1_2",
    "age_3_10",
    "age_11_100",
    "age_101_plus",
    "profit_ratio",
    "avg_cost_close_ratio",
    "peak_price_close_ratio",
    "avg_chip_age",
    "turnover"
]

RET_COLS = [
    "future_5d_ret",
    "future_10d_ret",
    "future_20d_ret"
]


# ================= 2. 横截面 RankIC 计算 =================

def calc_daily_rankic(group, factor_col, ret_col):
    """
    对某一天的横截面数据计算 RankIC。
    本质：corr(rank(factor), rank(future_return))
    """
    sub = group[[factor_col, ret_col]].replace([np.inf, -np.inf], np.nan).dropna()

    if len(sub) < MIN_STOCKS_PER_DAY:
        return np.nan

    factor_rank = sub[factor_col].rank()
    ret_rank = sub[ret_col].rank()

    return factor_rank.corr(ret_rank)


def calc_all_rankic(df):
    records = []

    grouped = df.groupby("trade_date")

    for trade_date, group in grouped:
        stock_count = group["ts_code"].nunique()

        for factor_col in FACTOR_COLS:
            for ret_col in RET_COLS:
                rankic = calc_daily_rankic(group, factor_col, ret_col)

                records.append({
                    "trade_date": trade_date,
                    "factor": factor_col,
                    "ret_col": ret_col,
                    "rankic": rankic,
                    "stock_count": stock_count
                })

    rankic_df = pd.DataFrame(records)
    rankic_df = rankic_df.dropna(subset=["rankic"]).reset_index(drop=True)

    return rankic_df


# ================= 3. 汇总统计 =================

def summarize_rankic(rankic_df):
    summary_records = []

    for (factor, ret_col), group in rankic_df.groupby(["factor", "ret_col"]):
        values = group["rankic"].dropna()

        if len(values) == 0:
            continue

        mean_rankic = values.mean()
        std_rankic = values.std()
        rankic_ir = mean_rankic / std_rankic if std_rankic != 0 else np.nan

        positive_ratio = (values > 0).mean()
        t_stat = mean_rankic / (std_rankic / np.sqrt(len(values))) if std_rankic != 0 else np.nan

        summary_records.append({
            "factor": factor,
            "ret_col": ret_col,
            "rankic_mean": mean_rankic,
            "rankic_std": std_rankic,
            "rankic_ir": rankic_ir,
            "rankic_t_stat": t_stat,
            "positive_ratio": positive_ratio,
            "sample_days": len(values)
        })

    summary_df = pd.DataFrame(summary_records)
    summary_df = summary_df.sort_values(
        ["ret_col", "rankic_mean"],
        ascending=[True, False]
    ).reset_index(drop=True)

    return summary_df


# ================= 4. 画累计 RankIC 曲线 =================

def plot_cumsum_rankic(rankic_df, ret_col="future_10d_ret"):
    plot_df = rankic_df[rankic_df["ret_col"] == ret_col].copy()

    pivot = plot_df.pivot(
        index="trade_date",
        columns="factor",
        values="rankic"
    ).sort_index()

    cumsum = pivot.cumsum()

    plt.figure(figsize=(14, 7))

    for factor in FACTOR_COLS:
        if factor in cumsum.columns:
            plt.plot(cumsum.index, cumsum[factor], label=factor)

    plt.title(f"Cumulative RankIC of CSI1000 Chip-Age Factors ({ret_col})")
    plt.xlabel("date")
    plt.ylabel("cumulative RankIC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_PATH, dpi=200)
    plt.close()

    print("累计 RankIC 图已保存：", OUTPUT_FIG_PATH)


# ================= 5. 主程序 =================

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"找不到输入文件：{INPUT_PATH}")

    print("正在读取因子数据：", INPUT_PATH)

    df = pd.read_csv(INPUT_PATH)
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    print("股票数量：", df["ts_code"].nunique())
    print("交易日数量：", df["trade_date"].nunique())
    print("总行数：", len(df))

    print("\n正在计算每日横截面 RankIC...")

    rankic_df = calc_all_rankic(df)

    rankic_df.to_csv(
        OUTPUT_DAILY_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("每日 RankIC 已保存：", OUTPUT_DAILY_PATH)

    print("\n正在汇总 RankIC 指标...")

    summary_df = summarize_rankic(rankic_df)

    summary_df.to_csv(
        OUTPUT_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("RankIC 汇总表已保存：", OUTPUT_SUMMARY_PATH)

    print("\n========== RankIC 汇总结果 ==========")
    print(summary_df)

    print("\n========== future_10d_ret 重点结果 ==========")
    print(
        summary_df[summary_df["ret_col"] == "future_10d_ret"]
        .sort_values("rankic_mean", ascending=False)
    )

    plot_cumsum_rankic(rankic_df, ret_col="future_10d_ret")