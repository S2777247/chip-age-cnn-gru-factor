import os
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# =========================
# 文件路径
# =========================

TENSOR_PATH = "data/csi1000_chip_tensor_n300.npy"
LABEL_PATH = "data/csi1000_label_10d_n300.npy"
DATE_PATH = "data/csi1000_tensor_dates_n300.csv"
STOCK_PATH = "data/csi1000_tensor_stocks_n300.csv"

# 可交易性过滤后的样本表
TRADABLE_PATH = "data/csi1000_chip_age_factor_n300_tradable.csv"

PRED_PATH = "data/cnn_gru_predictions_n300_rolling_ic_tradable.csv"
RANKIC_PATH = "data/cnn_gru_rankic_n300_rolling_ic_tradable.csv"
RANKIC_SUMMARY_PATH = "data/cnn_gru_rankic_summary_n300_rolling_ic_tradable.csv"
FIG_PATH = "data/cnn_gru_cumulative_rankic_n300_rolling_ic_tradable.png"


# =========================
# 参数设置
# =========================

WINDOW = 30

# 由于本文样本期为 2020-2026，无法使用原研报10年训练窗口；
# 这里采用约3年训练、半年验证、半年测试的滚动方式。
TRAIN_DAYS = 756
VAL_DAYS = 126
TEST_DAYS = 126
STEP_DAYS = 126
GAP_DAYS = 10

EPOCHS_PER_WINDOW = 3
LR = 1e-3
WEIGHT_DECAY = 1e-4
MIN_STOCKS_PER_DAY = 30

RANDOM_SEED = 42


# =========================
# 随机种子与设备
# =========================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# =========================
# 构造可交易性 mask
# =========================

def build_tradable_mask(dates, stocks):
    print("读取可交易性过滤样本：", TRADABLE_PATH)

    tradable_df = pd.read_csv(TRADABLE_PATH)

    tradable_df["trade_date"] = pd.to_datetime(
        tradable_df["trade_date"]
    ).dt.strftime("%Y-%m-%d")

    tradable_df["ts_code"] = tradable_df["ts_code"].astype(str)

    date_to_idx = {d: i for i, d in enumerate(dates)}
    stock_to_idx = {s: i for i, s in enumerate(stocks)}

    tradable_mask = np.zeros((len(dates), len(stocks)), dtype=bool)

    hit = 0

    for _, row in tradable_df.iterrows():
        d = row["trade_date"]
        s = row["ts_code"]

        if d in date_to_idx and s in stock_to_idx:
            tradable_mask[date_to_idx[d], stock_to_idx[s]] = True
            hit += 1

    print("可交易样本匹配数量：", hit)
    print("tradable_mask True 数量：", tradable_mask.sum())

    return tradable_mask


# =========================
# 模型
# =========================

class CNNGRUModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=4,
                out_channels=32,
                kernel_size=5,
                padding=2
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.gru = nn.GRU(
            input_size=32,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )

        self.head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: [stock_count, time, channel, price_bin]
        stock_count, time_steps, channels, price_bins = x.shape

        x = x.reshape(stock_count * time_steps, channels, price_bins)

        feat = self.cnn(x)
        feat = feat.squeeze(-1)

        feat = feat.reshape(stock_count, time_steps, -1)

        gru_out, _ = self.gru(feat)

        last_hidden = gru_out[:, -1, :]

        pred = self.head(last_hidden).squeeze(-1)

        return pred


# =========================
# IC loss
# =========================

def ic_loss(pred, target, eps=1e-8):
    """
    IC = corr(pred, target)
    loss = -IC
    训练时最小化 -IC，相当于最大化预测值与未来收益的横截面相关性。
    """
    pred = pred - pred.mean()
    target = target - target.mean()

    pred_std = torch.sqrt(torch.mean(pred ** 2) + eps)
    target_std = torch.sqrt(torch.mean(target ** 2) + eps)

    ic = torch.mean(pred * target) / (pred_std * target_std + eps)

    return -ic


# =========================
# 每个交易日构造横截面 batch
# =========================

def get_day_batch(t, tensor, label, tradable_mask):
    """
    返回某个交易日 t 的横截面样本：
    X: [当天可交易股票数, 30, 4, 32]
    y: [当天可交易股票数]
    """
    if t < WINDOW:
        return None, None, None

    stock_mask = tradable_mask[t].copy()

    y_all = label[t]
    x_all = tensor[t - WINDOW:t]  # [30, stock, 4, 32]
    x_all = np.transpose(x_all, (1, 0, 2, 3))  # [stock, 30, 4, 32]

    valid_y = np.isfinite(y_all)
    valid_x = np.isfinite(x_all).all(axis=(1, 2, 3))

    valid = stock_mask & valid_y & valid_x

    stock_indices = np.where(valid)[0]

    if len(stock_indices) < MIN_STOCKS_PER_DAY:
        return None, None, None

    x = x_all[stock_indices]
    y = y_all[stock_indices]

    # 如果当天未来收益几乎没有横截面差异，IC loss 没意义
    if np.nanstd(y) < 1e-8:
        return None, None, None

    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    return x_tensor, y_tensor, stock_indices


def valid_date_indices(date_range, tensor, label, tradable_mask):
    valid_dates = []

    for t in date_range:
        x, y, _ = get_day_batch(t, tensor, label, tradable_mask)

        if x is not None:
            valid_dates.append(t)

    return valid_dates


# =========================
# 训练、验证、预测
# =========================

def train_one_window(model, optimizer, train_dates, tensor, label, tradable_mask, device):
    model.train()

    random.shuffle(train_dates)

    losses = []

    for t in train_dates:
        x, y, _ = get_day_batch(t, tensor, label, tradable_mask)

        if x is None:
            continue

        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = ic_loss(pred, y)

        if torch.isnan(loss):
            continue

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    if len(losses) == 0:
        return np.nan

    return float(np.mean(losses))


def eval_window_ic(model, eval_dates, tensor, label, tradable_mask, device):
    model.eval()

    ics = []

    with torch.no_grad():
        for t in eval_dates:
            x, y, _ = get_day_batch(t, tensor, label, tradable_mask)

            if x is None:
                continue

            x = x.to(device)
            y = y.to(device)

            pred = model(x)

            loss = ic_loss(pred, y)

            if torch.isnan(loss):
                continue

            ic = -loss.item()
            ics.append(ic)

    if len(ics) == 0:
        return np.nan

    return float(np.mean(ics))


def predict_window(model, test_dates, tensor, label, tradable_mask, dates, stocks, device, roll_id):
    model.eval()

    records = []

    with torch.no_grad():
        for t in test_dates:
            x, y, stock_indices = get_day_batch(t, tensor, label, tradable_mask)

            if x is None:
                continue

            x = x.to(device)
            pred = model(x).cpu().numpy()
            y_np = y.numpy()

            for p, yy, s_idx in zip(pred, y_np, stock_indices):
                records.append({
                    "roll_id": roll_id,
                    "trade_date": dates[t],
                    "ts_code": stocks[s_idx],
                    "pred": float(p),
                    "future_10d_ret": float(yy)
                })

    return records


# =========================
# RankIC 统计
# =========================

def calc_rankic(pred_df):
    records = []

    for trade_date, group in pred_df.groupby("trade_date"):
        sub = group[["pred", "future_10d_ret"]].replace(
            [np.inf, -np.inf],
            np.nan
        ).dropna()

        if len(sub) < MIN_STOCKS_PER_DAY:
            continue

        rankic = sub["pred"].rank().corr(sub["future_10d_ret"].rank())

        records.append({
            "trade_date": trade_date,
            "rankic": rankic,
            "stock_count": len(sub)
        })

    return pd.DataFrame(records)


# =========================
# 主程序
# =========================

if __name__ == "__main__":
    set_seed(RANDOM_SEED)
    os.makedirs("data", exist_ok=True)

    device = get_device()
    print("使用设备：", device)

    print("读取 CNN-GRU 输入张量：", TENSOR_PATH)
    tensor = np.load(TENSOR_PATH)

    print("读取标签：", LABEL_PATH)
    label = np.load(LABEL_PATH)

    dates = pd.read_csv(DATE_PATH)["trade_date"].astype(str).tolist()
    stocks = pd.read_csv(STOCK_PATH)["ts_code"].astype(str).tolist()

    print("tensor shape:", tensor.shape)
    print("label shape:", label.shape)
    print("交易日数量：", len(dates))
    print("股票数量：", len(stocks))

    tradable_mask = build_tradable_mask(dates, stocks)

    all_pred_records = []
    roll_summaries = []

    num_dates = len(dates)

    # 有效样本通常从筹码预热期后才开始
    roll_start = WINDOW

    roll_id = 0

    while True:
        train_start = roll_start
        train_end = train_start + TRAIN_DAYS

        val_start = train_end + GAP_DAYS
        val_end = val_start + VAL_DAYS

        test_start = val_end + GAP_DAYS
        test_end = test_start + TEST_DAYS

        if test_end >= num_dates - 10:
            break

        train_range = range(train_start, train_end)
        val_range = range(val_start, val_end)
        test_range = range(test_start, test_end)

        train_dates = valid_date_indices(train_range, tensor, label, tradable_mask)
        val_dates = valid_date_indices(val_range, tensor, label, tradable_mask)
        test_dates = valid_date_indices(test_range, tensor, label, tradable_mask)

        if len(train_dates) < 50 or len(val_dates) < 20 or len(test_dates) < 20:
            roll_start += STEP_DAYS
            continue

        roll_id += 1

        print("\n" + "=" * 60)
        print(f"滚动窗口 {roll_id}")
        print("训练区间：", dates[train_dates[0]], "至", dates[train_dates[-1]], "交易日数：", len(train_dates))
        print("验证区间：", dates[val_dates[0]], "至", dates[val_dates[-1]], "交易日数：", len(val_dates))
        print("测试区间：", dates[test_dates[0]], "至", dates[test_dates[-1]], "交易日数：", len(test_dates))

        model = CNNGRUModel().to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY
        )

        best_val_ic = -math.inf
        best_state = None

        for epoch in range(1, EPOCHS_PER_WINDOW + 1):
            train_loss = train_one_window(
                model,
                optimizer,
                train_dates,
                tensor,
                label,
                tradable_mask,
                device
            )

            val_ic = eval_window_ic(
                model,
                val_dates,
                tensor,
                label,
                tradable_mask,
                device
            )

            print(
                f"Epoch {epoch}/{EPOCHS_PER_WINDOW} | "
                f"train_loss={train_loss:.6f} | "
                f"val_ic={val_ic:.6f}"
            )

            if np.isfinite(val_ic) and val_ic > best_val_ic:
                best_val_ic = val_ic
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }

        if best_state is not None:
            model.load_state_dict(best_state)

        test_ic = eval_window_ic(
            model,
            test_dates,
            tensor,
            label,
            tradable_mask,
            device
        )

        print(f"窗口 {roll_id} 验证集最佳IC：{best_val_ic:.6f}")
        print(f"窗口 {roll_id} 测试集平均IC：{test_ic:.6f}")

        pred_records = predict_window(
            model,
            test_dates,
            tensor,
            label,
            tradable_mask,
            dates,
            stocks,
            device,
            roll_id
        )

        all_pred_records.extend(pred_records)

        roll_summaries.append({
            "roll_id": roll_id,
            "train_start": dates[train_dates[0]],
            "train_end": dates[train_dates[-1]],
            "val_start": dates[val_dates[0]],
            "val_end": dates[val_dates[-1]],
            "test_start": dates[test_dates[0]],
            "test_end": dates[test_dates[-1]],
            "train_days": len(train_dates),
            "val_days": len(val_dates),
            "test_days": len(test_dates),
            "best_val_ic": best_val_ic,
            "test_ic_mean": test_ic
        })

        roll_start += STEP_DAYS

    if len(all_pred_records) == 0:
        raise RuntimeError("没有生成任何预测结果，请检查滚动窗口参数。")

    pred_df = pd.DataFrame(all_pred_records)

    pred_df.to_csv(
        PRED_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    rankic_df = calc_rankic(pred_df)

    rankic_df.to_csv(
        RANKIC_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    mean_rankic = rankic_df["rankic"].mean()
    std_rankic = rankic_df["rankic"].std()
    rankic_ir = mean_rankic / std_rankic if std_rankic != 0 else np.nan
    positive_ratio = (rankic_df["rankic"] > 0).mean()
    t_stat = (
        mean_rankic / std_rankic * np.sqrt(len(rankic_df))
        if std_rankic != 0
        else np.nan
    )

    summary_df = pd.DataFrame([{
        "rankic_mean": mean_rankic,
        "rankic_std": std_rankic,
        "rankic_ir": rankic_ir,
        "rankic_t_stat": t_stat,
        "positive_ratio": positive_ratio,
        "sample_days": len(rankic_df),
        "rolling_windows": len(roll_summaries)
    }])

    summary_df.to_csv(
        RANKIC_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(roll_summaries).to_csv(
        "data/cnn_gru_rolling_window_summary_n300_ic_tradable.csv",
        index=False,
        encoding="utf-8-sig"
    )

    rankic_df["trade_date"] = pd.to_datetime(rankic_df["trade_date"])
    rankic_df = rankic_df.sort_values("trade_date")
    rankic_df["cumulative_rankic"] = rankic_df["rankic"].cumsum()

    plt.figure(figsize=(10, 5))
    plt.plot(
        rankic_df["trade_date"],
        rankic_df["cumulative_rankic"],
        label="CNN-GRU rolling IC"
    )
    plt.title("Cumulative RankIC of CNN-GRU Factor - Rolling IC Loss")
    plt.xlabel("date")
    plt.ylabel("cumulative RankIC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=300)
    plt.close()

    print("\n========== CNN-GRU 滚动训练 + IC Loss 结果 ==========")
    print(summary_df)

    print("\n预测结果保存：", PRED_PATH)
    print("RankIC明细保存：", RANKIC_PATH)
    print("RankIC汇总保存：", RANKIC_SUMMARY_PATH)
    print("滚动窗口汇总保存：data/cnn_gru_rolling_window_summary_n300_ic_tradable.csv")
    print("累计RankIC图保存：", FIG_PATH)