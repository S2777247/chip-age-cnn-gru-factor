import os
import time
import pandas as pd
import tushare as ts
from tushare.pro import client as _ts_client

_ts_client.DataApi._DataApi__http_url = "https://tu.brze.top"

TOKEN = os.getenv("TUSHARE_TOKEN") or input("请输入你的 Tushare Token：").strip()
pro = ts.pro_api(TOKEN)

TS_CODE = "002594.SZ"
START_DATE = "20200101"
END_DATE = "20260423"

# 读取刚才生成的筹码龄占比数据
ratio_df = pd.read_csv(f"chip_age_ratio_{TS_CODE}.csv")
ratio_df["trade_date"] = pd.to_datetime(ratio_df["trade_date"])

# 拉取收盘价
daily = pro.daily(
    ts_code=TS_CODE,
    start_date=START_DATE,
    end_date=END_DATE
)

time.sleep(0.6)

adj = pro.adj_factor(
    ts_code=TS_CODE,
    start_date=START_DATE,
    end_date=END_DATE
)

df = daily.merge(adj, on=["ts_code", "trade_date"], how="left")
df = df.sort_values("trade_date").reset_index(drop=True)

df["trade_date"] = pd.to_datetime(df["trade_date"])
df["close"] = pd.to_numeric(df["close"], errors="coerce")
df["adj_factor"] = pd.to_numeric(df["adj_factor"], errors="coerce")

latest_adj = df["adj_factor"].iloc[-1]
df["close_qfq"] = df["close"] * df["adj_factor"] / latest_adj

price_df = df[["trade_date", "close_qfq"]].copy()

# 合并筹码因子和价格数据
factor_df = ratio_df.merge(price_df, on="trade_date", how="left")

# 计算未来收益
factor_df["future_5d_ret"] = factor_df["close_qfq"].shift(-5) / factor_df["close_qfq"] - 1
factor_df["future_10d_ret"] = factor_df["close_qfq"].shift(-10) / factor_df["close_qfq"] - 1
factor_df["future_20d_ret"] = factor_df["close_qfq"].shift(-20) / factor_df["close_qfq"] - 1

factor_df = factor_df.dropna()

factor_cols = [
    "age_1_2",
    "age_3_10",
    "age_11_100",
    "age_101_plus"
]

print("========== 单股筹码龄因子检验 ==========")

for col in factor_cols:
    corr_5 = factor_df[col].corr(factor_df["future_5d_ret"], method="spearman")
    corr_10 = factor_df[col].corr(factor_df["future_10d_ret"], method="spearman")
    corr_20 = factor_df[col].corr(factor_df["future_20d_ret"], method="spearman")

    print("\n因子：", col)
    print("未来5日收益 Spearman相关：", corr_5)
    print("未来10日收益 Spearman相关：", corr_10)
    print("未来20日收益 Spearman相关：", corr_20)

factor_df.to_csv(f"single_stock_factor_test_{TS_CODE}.csv", index=False, encoding="utf-8-sig")
print("\n已保存：", f"single_stock_factor_test_{TS_CODE}.csv")

