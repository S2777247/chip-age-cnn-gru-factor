import os
import time
import pandas as pd
import tushare as ts
from tushare.pro import client as _ts_client

# ================= 1. Tushare 设置 =================

_ts_client.DataApi._DataApi__http_url = "https://tu.brze.top"

TOKEN = os.getenv("TUSHARE_TOKEN") or input("请输入你的 Tushare Token：").strip()
pro = ts.pro_api(TOKEN)


# ================= 2. 参数设置 =================

INDEX_CODE = "000852.SH"      # 中证1000指数代码
START_DATE = "20200101"
END_DATE = "20260423"

MAX_STOCKS = 300              # 这次拉300只

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


# ================= 3. 获取中证1000成分股 =================

def get_csi1000_stocks(index_code, start_date, end_date, max_stocks=300):
    print("正在获取中证1000成分股...")

    comp = pro.index_weight(
        index_code=index_code,
        start_date=start_date,
        end_date=end_date
    )

    if comp.empty:
        raise ValueError

    comp = comp.sort_values("trade_date")

    latest_date = comp["trade_date"].max()
    latest_comp = comp[comp["trade_date"] == latest_date].copy()

    latest_comp = latest_comp.sort_values("weight", ascending=False)
    latest_comp = latest_comp.drop_duplicates(subset=["con_code"])

    if max_stocks is not None:
        latest_comp = latest_comp.head(max_stocks)

    stocks = latest_comp["con_code"].tolist()

    print("成分股截面日期：", latest_date)
    print("本次股票数量：", len(stocks))
    print("前10只股票：", stocks[:10])

    comp_path = f"{DATA_DIR}/csi1000_components_{latest_date}_n{len(stocks)}.csv"
    latest_comp.to_csv(comp_path, index=False, encoding="utf-8-sig")

    print("成分股文件已保存：", comp_path)

    return stocks, latest_date


# ================= 4. 拉取单只股票数据 =================

def get_one_stock_data(ts_code, start_date, end_date):
    daily = pro.daily(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date
    )
    time.sleep(0.45)

    basic = pro.daily_basic(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
        fields="ts_code,trade_date,turnover_rate,turnover_rate_f,total_mv,circ_mv"
    )
    time.sleep(0.45)

    adj = pro.adj_factor(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date
    )
    time.sleep(0.45)

    if daily.empty:
        return pd.DataFrame()

    df = daily.merge(
        basic,
        on=["ts_code", "trade_date"],
        how="left"
    )

    df = df.merge(
        adj,
        on=["ts_code", "trade_date"],
        how="left"
    )

    df = df.sort_values("trade_date").reset_index(drop=True)

    return df


# ================= 5. 批量拉取300只股票 =================

def get_stock_pool_data(stocks, start_date, end_date):
    all_data = []
    failed = []

    for i, ts_code in enumerate(stocks, start=1):
        print(f"[{i}/{len(stocks)}] 正在拉取：{ts_code}")

        try:
            df = get_one_stock_data(ts_code, start_date, end_date)

            if df.empty:
                print(f"  无数据：{ts_code}")
                failed.append(ts_code)
                continue

            all_data.append(df)
            print(f"  成功：{len(df)} 行")

        except Exception as e:
            print(f"  失败：{ts_code}")
            print(f"  原因：{e}")
            failed.append(ts_code)

        # 防止请求太快
        time.sleep(0.6)

        # 每50只临时保存一次，防止中途断了全没了
        if i % 50 == 0 and len(all_data) > 0:
            temp_df = pd.concat(all_data, ignore_index=True)
            temp_path = f"{DATA_DIR}/temp_csi1000_daily_n{len(all_data)}.csv"
            temp_df.to_csv(temp_path, index=False, encoding="utf-8-sig")
            print(f"  临时保存：{temp_path}")

    if len(all_data) == 0:
        raise ValueError("没有成功获取任何股票数据。")

    result = pd.concat(all_data, ignore_index=True)

    failed_df = pd.DataFrame({"failed_ts_code": failed})
    failed_path = f"{DATA_DIR}/failed_stocks_n300.csv"
    failed_df.to_csv(failed_path, index=False, encoding="utf-8-sig")

    print("失败股票数量：", len(failed))
    print("失败股票文件：", failed_path)

    return result


# ================= 6. 数据清洗 =================

def clean_data(df):
    numeric_cols = [
        "open", "high", "low", "close", "pre_close",
        "change", "pct_chg", "vol", "amount",
        "turnover_rate", "turnover_rate_f",
        "total_mv", "circ_mv", "adj_factor"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["trade_date"] = pd.to_datetime(df["trade_date"])

    # 优先使用自由流通换手率
    df["turnover"] = df["turnover_rate_f"].fillna(df["turnover_rate"]) / 100

    # 防止极端换手率破坏筹码递推
    df["turnover"] = df["turnover"].clip(lower=0, upper=0.95)

    # 删除核心字段缺失行
    df = df.dropna(subset=[
        "ts_code",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "vol",
        "amount",
        "adj_factor",
        "turnover"
    ])

    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    return df


# ================= 7. 主程序 =================

if __name__ == "__main__":
    stocks, component_date = get_csi1000_stocks(
        index_code=INDEX_CODE,
        start_date=START_DATE,
        end_date=END_DATE,
        max_stocks=MAX_STOCKS
    )

    raw_df = get_stock_pool_data(
        stocks=stocks,
        start_date=START_DATE,
        end_date=END_DATE
    )

    clean_df = clean_data(raw_df)

    output_path = f"{DATA_DIR}/csi1000_daily_data_{START_DATE}_{END_DATE}_n{MAX_STOCKS}.csv"

    clean_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n========== 数据获取完成 ==========")
    print("成分股截面日期：", component_date)
    print("股票数量：", clean_df["ts_code"].nunique())
    print("交易日数量：", clean_df["trade_date"].nunique())
    print("总行数：", len(clean_df))
    print("保存路径：", output_path)

    print("\n前5行：")
    print(clean_df.head())
