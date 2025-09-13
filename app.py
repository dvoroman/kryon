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

EXCHANGE_ID = CFG.get("exchange_id", "binanceusdm")
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

exchange = getattr(ccxt, EXCHANGE_ID)({
    "enableRateLimit": True,
})

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
    atr = ta.volatility.AverageTrueRange(d["high"], d["low"], d["close"], window=14)
    d["atr"] = atr.average_true_range()
    d["ema20"] = ta.trend.EMAIndicator(d["close"], window=20).ema_indicator()
    d["ema50"] = ta.trend.EMAIndicator(d["close"], window=50).ema_indicator()
    d["ema200"] = ta.trend.EMAIndicator(d["close"], window=200).ema_indicator()
    d["ema20_50"] = d["ema20"] - d["ema50"]
    d["ema50_200"] = d["ema50"] - d["ema200"]
    bb = ta.volatility.BollingerBands(d["close"], window=20, window_dev=2)
    d["bb_high"] = bb.bollinger_hband()
    d["bb_low"] = bb.bollinger_lband()
    d["bb_width"] = (d["bb_high"] - d["bb_low"]) / d["close"]
    d["adx"] = ta.trend.ADXIndicator(d["high"], d["low"], d["close"], window=14).adx()
    d["rsi"] = ta.momentum.RSIIndicator(d["close"], window=14).rsi()
    d["vol_z"] = (d["volume"] - d["volume"].rolling(50).mean()) / (d["volume"].rolling(50).std() + 1e-9)
    d["dev_ema20"] = (d["close"] - d["ema20"]) / (d["atr"] + 1e-9)
    return d

def detect_regime(ctx: pd.DataFrame) -> str:
    c = ctx.dropna().copy()
    if len(c) < 80:
        return "unknown"
    last = c.iloc[-1]
    ema_slope = (c["ema50"].iloc[-1] - c["ema50"].iloc[-10]) / 10.0
    adx = last["adx"]
    # approximate ATR percentile
    atr_pct = (c["atr"] / c["close"]).rolling(100).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1]).iloc[-1]
    if adx > 20 and ema_slope > 0:
        return "trend_up"
    if adx > 20 and ema_slope < 0:
        return "trend_down"
    if atr_pct > 0.85 and c["ret6"].iloc[-1] < -0.03:
        return "panic_down"
    return "range"

def triple_barrier_labels(d: pd.DataFrame, tp_k=1.5, sl_k=1.0, horizon=48):
    n = len(d)
    y = np.zeros(n)
    for i in range(n - horizon - 1):
        c0 = d["close"].iloc[i]
        atr0 = d["atr"].iloc[i]
        up = c0 + tp_k * atr0
        dn = c0 - sl_k * atr0
        seg = d.iloc[i+1:i+1+horizon]
        hit_up = (seg["high"] >= up).idxmax() if (seg["high"] >= up).any() else None
        hit_dn = (seg["low"] <= dn).idxmax() if (seg["low"] <= dn).any() else None
        if hit_up is not None and hit_dn is not None:
            if seg.index.get_loc(hit_up) < seg.index.get_loc(hit_dn):
                y[i] = 1
            else:
                y[i] = 2
        elif hit_up is not None:
            y[i] = 1
        elif hit_dn is not None:
            y[i] = 2
        else:
            y[i] = 0
    return y

def train_model(feat: pd.DataFrame, labels: np.ndarray):
    df = feat.dropna().copy()
    y = labels[-len(df):]
    y_bin = (y == 1).astype(int)
    X = df[[
        "ret1","ret3","ret6","ret12","ret24",
        "atr","ema20_50","ema50_200","bb_width","adx","rsi","vol_z","dev_ema20"
    ]].values
    if (y_bin.sum() == 0) or (y_bin.sum() == len(y_bin)):
        return None
    base = LogisticRegression(max_iter=200, class_weight='balanced')
    model = CalibratedClassifierCV(base_estimator=base, method='sigmoid', cv=TimeSeriesSplit(n_splits=3))
    model.fit(X, y_bin)
    return model, df.index

def latest_signal_prob(model, df_feat: pd.DataFrame):
    row = df_feat.dropna().iloc[-1:]
    if row.empty:
        return None
    Xlast = row[[
        "ret1","ret3","ret6","ret12","ret24",
        "atr","ema20_50","ema50_200","bb_width","adx","rsi","vol_z","dev_ema20"
    ]].values
    p_up = float(model.predict_proba(Xlast)[0,1])
    return p_up

def expected_value(p: float, R: float, costs: float = 0.0015):
    return p*R - (1-p) - costs

last_signal_ts = {}
def already_cooled(symbol: str, minutes: int) -> bool:
    ts = last_signal_ts.get(symbol)
    if ts is None:
        return True
    return (time.time() - ts) > (minutes * 60)

def mark_signal(symbol: str):
    last_signal_ts[symbol] = time.time()

def format_signal(side: str, symbol: str, price: float, atr: float, regime: str, p: float, R: float):
    if side == 'LONG':
        e1 = price - 0.20*atr
        e2 = price + 0.10*atr
        sl = price - 1.0*atr
        tp1 = price + 1.0*atr
        tp2 = price + 2.0*atr
        tp3 = price + 3.0*atr
    else:
        e1 = price + 0.20*atr
        e2 = price - 0.10*atr
        sl = price + 1.0*atr
        tp1 = price - 1.0*atr
        tp2 = price - 2.0*atr
        tp3 = price - 3.0*atr

    lines = [
        f"#{side} {symbol} (PERP) 5x",
        f"Entry:    {e1:.2f} – {e2:.2f}",
        f"SL:       {sl:.2f}",
        f"TP1:      {tp1:.2f} (40%)",
        f"TP2:      {tp2:.2f} (35%)",
        f"TP3:      {tp3:.2f} (25%)",
        "",
        f"Confidence: {p:.2f}",
        f"R/R:       {R:.2f}",
        f"Regime:    {regime}",
        f"Notes:     Auto‑generated (ATR/EMA/BB/ADX/vol)"
    ]
    return "\n".join(lines)

def scan_once():
    for symbol in PAIRS:
        try:
            if not already_cooled(symbol, cooldown_minutes):
                logging.info(f"{symbol}: cooldown")
                continue
            df = fetch_df(symbol, TF, LIMIT)
            df = add_features(df)
            ctx = fetch_df(symbol, CTX_TF, LIMIT)
            ctx = add_features(ctx)
            regime = detect_regime(ctx)

            labels = triple_barrier_labels(df, TP_ATR, SL_ATR, HZN)
            feat = df.copy()
            model_out = train_model(feat, labels)
            if model_out is None:
                logging.info(f"{symbol}: model not trained")
                continue
            model, idx = model_out
            p_up = latest_signal_prob(model, feat)
            if p_up is None:
                continue

            last = df.dropna().iloc[-1]
            price = float(last["close"])
            atr = float(last["atr"])
            adx = float(last["adx"])
            volz = float(last["vol_z"])

            p_down = 1.0 - p_up
            R = TP_ATR / SL_ATR if SL_ATR > 0 else 1.5

            side = None
            if p_up >= p_min and volz >= -0.2:
                if regime in ("trend_up", "range") and adx >= 12:
                    side = 'LONG'
            if p_down >= p_min and volz >= -0.2:
                if regime in ("trend_down", "range") and adx >= 12:
                    if side is None or p_down > p_up:
                        side = 'SHORT'

            if side is None:
                logging.info(f"{symbol}: gating filtered (p_up={p_up:.2f}, regime={regime}, adx={adx:.1f}, volz={volz:.2f})")
                continue

            p = p_up if side=='LONG' else p_down
            ev = expected_value(p, R, costs=0.0015)
            if p < p_min or ev < ev_min:
                logging.info(f"{symbol}: EV/p filter (p={p:.2f}, EV={ev:.3f})")
                continue

            text = format_signal(side, symbol.replace(":USDT",""), price, atr, regime, p, R)
            logging.info(f"SIGNAL {symbol}: side={side} p={p:.2f} EV={ev:.3f}")
            if not dry_run:
                tg_send_text(f"```\n{text}\n```", markdown=True)
            mark_signal(symbol)
        except Exception as e:
            logging.error(f"Error {symbol}: {e}\n{traceback.format_exc()}")

if __name__ == "__main__":
    # single run by default
    scan_once()
