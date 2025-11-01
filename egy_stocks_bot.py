# -*- coding: utf-8 -*-
import os
import threading
import logging
from typing import Tuple

import pandas as pd
import numpy as np
import yfinance as yf

from flask import Flask
from telegram import Update, ParseMode
from telegram.ext import Updater, CommandHandler, CallbackContext

# ==================== إعدادات عامة ====================
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("EGY_STOCKS_BOT")

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")  # لازم تضيفه في Render
if not BOT_TOKEN:
    raise RuntimeError("Environment variable TELEGRAM_BOT_TOKEN is missing!")

# ==================== وظائف مساعدة ====================
def egx_ticker(text: str) -> str:
    """يرجع الرمز بصيغة EGX (يضيف .CA لو ناقصة)"""
    t = (text or "").strip().upper()
    if not t.endswith(".CA"):
        t = f"{t}.CA"
    return t

def fetch_ohlc(ticker: str, period="9mo", interval="1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if isinstance(df, pd.DataFrame) and not df.empty:
        return df.dropna().copy()
    return pd.DataFrame()

def swing_levels(series: pd.Series, lookback: int = 20) -> Tuple[float, float]:
    recent = series[-lookback:]
    return float(recent.min()), float(recent.max())

def analyze_symbol(ticker_raw: str) -> str:
    ticker = egx_ticker(ticker_raw)
    df = fetch_ohlc(ticker)
    if df.empty or len(df) < 30:
        return f"⚠️ لا توجد بيانات كافية للرمز: {ticker_raw}.\nجرّب رمزًا مختلفًا."

    close = df["Close"]
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()

    current_price = float(close.iloc[-1])
    sup, res = swing_levels(close, 25)

    # نص الاتجاهات
    def trend_text(fast, slow):
        if np.isnan(fast) or np.isnan(slow):
            return "غير واضح"
        return "صاعد" if fast > slow else "هابط"

    t_short = trend_text(sma20.iloc[-1], sma50.iloc[-1])
    t_medium = trend_text(sma50.iloc[-1], close.rolling(100).mean().iloc[-1])

    # نقاط مقترحة مبسطة
    target1 = round(res, 2)
    target2 = round(res * 1.035, 2)
    stop    = round(sup * 0.985, 2)
    buy_break     = round(res * 1.002, 2)
    buy_pullback  = round(max(sup, sma20.iloc[-1]), 2)

    lines = []
    lines.append("📈 *EGY STOCKS BOT*")
    lines.append(f"الرمز: *{ticker_raw.upper()}*")
    lines.append(f"السعر الحالي: *{round(current_price,2)}*")
    lines.append("")
    lines.append("مستويات الدعم والمقاومة:")
    lines.append(f"• الدعم: *{round(sup,2)}*")
    lines.append(f"• المقاومة: *{round(res,2)}*")
    lines.append("")
    lines.append("الاتجاهات:")
    lines.append(f"• القصير: *{t_short}*")
    lines.append(f"• المتوسط: *{t_medium}*")
    lines.append("")
    lines.append("توصيات تداول مبسطة (تعليمية وليست نصيحة استثمارية):")
    lines.append(f"• شراء *تأكيد*: اختراق فوق *{buy_break}*")
    lines.append(f"• شراء *Pullback*: قرب *{buy_pullback}* (مع مراقبة الحجم)")
    lines.append(f"• الأهداف: *{target1}* ثم *{target2}*")
    lines.append(f"• وقف الخسارة: أسفل *{stop}*")

    return "\n".join(lines)

# ==================== أوامر البوت ====================
def start(update: Update, context: CallbackContext) -> None:
    txt = (
        "أهلًا 👋\n\n"
        "أنا بوت تحليلات سريعة لأسهم البورصة المصرية.\n"
        "استخدم:\n"
        "• /signal COMI  ← توصية سريعة (يمكن كتابة الرمز بدون .CA)\n"
        "• /help        ← المساعدة\n"
        "مثال: /signal EGAL"
    )
    update.message.reply_text(txt)

def help_cmd(update: Update, context: CallbackContext) -> None:
    update.message.reply_text(
        "الأوامر:\n"
        "• /start  للبدء\n"
        "• /signal <رمز> مثال: /signal EFIH"
    )

def signal_cmd(update: Update, context: CallbackContext) -> None:
    if len(context.args) == 0:
        update.message.reply_text("اكتب الرمز بعد الأمر، مثال: /signal COMI")
        return
    symbol = context.args[0]
    try:
        msg = analyze_symbol(symbol)
        update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logger.exception("signal error")
        update.message.reply_text(f"حصل خطأ غير متوقع: {e}")

# ==================== التشغيل ====================
def run_bot():
    updater = Updater(BOT_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help_cmd))
    dp.add_handler(CommandHandler("signal", signal_cmd, pass_args=True))

    updater.start_polling()
    updater.idle()

# سيرفر صغير لِـ Render (Web Service يحتاج بورت)
app = Flask(__name__)

@app.route("/")
def health():
    return "EGY Stocks Bot is running", 200

if __name__ == "__main__":
    # شغّل البوت في Thread
    t = threading.Thread(target=run_bot, daemon=True)
    t.start()

    # شغّل Flask على البورت الذي يقدمه Render
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
