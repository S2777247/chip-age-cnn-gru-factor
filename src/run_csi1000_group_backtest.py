import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


INPUT_PATH = "data/csi1000_chip_age_factor_n300.csv"
OUTPUT_GROUP_RET = "data/csi1000_group_return_n300.csv"
OUTPUT_SUMMARY = "data/csi1000_group_summary_n300.csv"
OUTPUT_FIG = "data/csi1000_group_nav_age_11_100_n300.png"

FACTOR = "age_11_100"
RET_COL = "future_10d_ret"
N_GROUPS = 5
MIN_STOCKS_PER_DAY = 50


def assign_groups(group, factor_col, n_groups):
    group = group.copy()
    group = group.replace([np.inf, -np.inf], np.nan)
    group = group.dropna(subset=[factor_col, RET_COL])

    if len(group) < MIN_STOCKS_PER_DAY:
        return pd.DataFrame()

    try:
        group["group"] = pd.qcut(
            group[factor_col],
            q=n_groups,
            labels=[f"G{i}" for i in range(1, n_groups + 1)],
            duplicates="drop"
        )
    except ValueError:
        return pd.DataFrame()

    return group


def run_group_backtest(df, factor_col, ret_col):
    records = []

    for trade_date, group in df.groupby("trade_date"):
        grouped = assign_groups(group, factor_col, N_GROUPS)

        if grouped.empty:
            continue

        group_ret = grouped.groupby("group", observed=False)[ret_col].mean()

        record = {"trade_date": trade_date}

        for g in [f"G{i}" for i in range(1, N_GROUPS + 1)]:
            record[g] = group_ret.get(g, np.nan)

        if "G1" in record and "G5" in record:
            record["long_short"] = record["G5"] - record["G1"]
        else:
            record["long_short"] = np.nan

        records.append(record)

    result = pd.DataFrame(records)
    result = result.sort_values("trade_date").reset_index(drop=True)

    return result


def summarize_group_return(group_df):
    summary = []

    ret_cols = [f"G{i}" for i in range(1, N_GROUPS + 1)] + ["long_short"]

    for col in ret_cols:
        values = group_df[col].dropna()

        if len(values) == 0:
            continue

        mean_ret = values.mean()
        std_ret = values.std()
        win_rate = (values > 0).mean()

        # 这里是10日持有收益的简单统计，不严格年化
        summary.append({
            "portfolio": col,
            "mean_10d_ret": mean_ret,
            "std_10d_ret": std_ret,
            "t_stat": mean_ret / (std_ret / np.sqrt(len(values))) if std_ret != 0 else np.nan,
            "win_rate": win_rate,
            "sample_days": len(values)
        })

    return pd.DataFrame(summary)


def plot_group_nav(group_df):
    plot_df = group_df.copy()
    plot_df["trade_date"] = pd.to_datetime(plot_df["trade_date"])

    ret_cols = [f"G{i}" for i in range(1, N_GROUPS + 1)] + ["long_short"]

    plt.figure(figsize=(14, 7))

    for col in ret_cols:
        if col in plot_df.columns:
            nav = (1 + plot_df[col].fillna(0)).cumprod()
            plt.plot(plot_df["trade_date"], nav, label=col)

    plt.title(f"Group NAV by {FACTOR} ({RET_COL})")
    plt.xlabel("date")
    plt.ylabel("cumulative NAV")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG, dpi=200)
    plt.close()

    print("分组净值图已保存：", OUTPUT_FIG)


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    print("读取数据：", INPUT_PATH)
    print("股票数量：", df["ts_code"].nunique())
    print("交易日数量：", df["trade_date"].nunique())
    print("总行数：", len(df))

    group_df = run_group_backtest(df, FACTOR, RET_COL)
    group_df.to_csv(OUTPUT_GROUP_RET, index=False, encoding="utf-8-sig")

    summary_df = summarize_group_return(group_df)
    summary_df.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8-sig")

    print("\n========== 分组收益汇总 ==========")
    print(summary_df)

    print("\n分组收益文件：", OUTPUT_GROUP_RET)
    print("分组汇总文件：", OUTPUT_SUMMARY)

    plot_group_nav(group_df)