import os, time, math, json, logging, traceback
import requests
import yaml
import numpy as np
import pandas as pd
import ccxt
from datetime import datetime, timezone

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit

import ta  # технические индикаторы

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHANNEL_ID = os.getenv("CHANNEL_ID")
TG_API = f"https://api.telegram.org/bot{BOT_TOKEN}"

def tg_send_text(text: str, markdown: bool = True):
    if CHANNEL_ID is None or BOT_TOKEN is None:
        logging.warning("No BOT_TOKEN/CHANNEL_ID; skip send")
        return
    payload = {
        "chat_id": CHANNEL_ID,
        "text": text,
        "disable_web_page_preview": True
    }
    if markdown:
        payload["parse_mode"] = "Markdown"
    try:
        r = requests.post(f"{TG_API}/sendMessage", json=payload, timeout=20)
        if r.status_code != 200:
            logging.error(f"TG send error: {r.status_code} {r.text}")
    except Exception as e:
        logging.error(f"TG send exception: {e}")

with open("config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

EXCHANGE_ID = CFG.get("exchange_id", "bybit")
PAIRS = CFG.get("pairs", ["BTC/USDT:USDT"])
TF = CFG.get("scan_timeframe", "15m")
CTX_TF = CFG.get("ctx_timeframe", "1h")
LIMIT = int(CFG.get("limit_bars", 600))

LBL = CFG.get("label", {})
TP_ATR = float(LBL.get("tp_atr", 1.5))
SL_ATR = float(LBL.get("sl_atr", 1.0))
HZN = int(LBL.get("horizon_bars", 48))

TH = CFG.get("thresholds", {})
p_min = float(TH.get("p_min", 0.58))
ev_min = float(TH.get("ev_min", 0.07))
cooldown_minutes = int(TH.get("cooldown_minutes", 360))

RISK = CFG.get("risk", {})
risk_per_trade_pct = float(RISK.get("risk_per_trade_pct", 0.7))
leverage = int(RISK.get("leverage", 5))

TGCFG = CFG.get("telegram", {})
send_markdown = bool(TGCFG.get("send_markdown", True))
dry_run = bool(TGCFG.get("dry_run", False))

# =========================
# CCXT exchange (Bybit/OKX → defaultType=swap)
# =========================
exchange_opts = {"enableRateLimit": True}
if EXCHANGE_ID.lower() in ["bybit", "okx"]:
    exchange_opts["options"] = {"defaultType": "swap"}

exchange = getattr(ccxt, EXCHANGE_ID)(exchange_opts)

# =========================
# Helpers
# =========================
def fetch_df(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=None, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ret1"] = np.log(d["close"]).diff(1)
    for k in [3,6,12,24]:
        d[f"ret{k}"] = np.log(d["close"]).diff(k)
    a
