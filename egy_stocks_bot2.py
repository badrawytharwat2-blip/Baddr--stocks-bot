# -*- coding: utf-8 -*-
"""
EGY Stocks Bot (PTB 13.15 compatible)
Author: Badrawy + GPT
Commands:
  /start                -> ترحيب وتعليمات
  /signal <TICKER>      -> يطلع لك توصية سريعة (مثال: /signal COMI.CA)
  /help                 -> مساعدة سريعة
Notes:
- يعتمد على yfinance لجلب البيانات (استخدم لاحقة .CA لأسهم البورصة المصرية على ياهو)
- صياغة الرسالة بالعربي وبنفس روح شاشة البوت اللي وريتنا صورتها
"""

import os
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator

from telegram import ParseMode
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

# ===================== إعداد اللوج =====================
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger("EGY-STOCKS-BOT")

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")


# ===================== أدوات مساعدة =====================
def egx_ticker(text: str) -> str:
    """ يحاول يضيف .CA لو المستخدم كتب كومى بدون لاحقة """
    t = text.strip().upper()
    if not t.endswith(".CA"):
        t = f"{t}.CA"
    return t


def fetch_ohlc(ticker: str, period="6mo", interval="1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if isinstance(df, pd.DataFrame) and not df.empty:
        df = df.dropna().copy()
    return df


def swing_levels(series: pd.Series, lookback: int = 20):
    """ ابسط دعم/مقاومة: أقل/أعلى قيمة لآخر نافذة """
    recent = series[-lookback:]
    return float(recent.min()), float(recent.max())


def trend_text(sma_fast: float, sma_slow: float) -> str:
    if np.isnan(sma_fast) or np.isnan(sma_slow):
        return "غير واضح"
    if sma_fast > sma_slow:
        return "صاعد"
    if sma_fast < sma_slow:
        return "هابط"
    return "محايد"


def build_signal_message(ticker_raw: str) -> str:
    ticker = egx_ticker(ticker_raw)
    df = fetch_ohlc(ticker, period="9mo", interval="1d")
    if df is None or df.empty or len(df) < 30:
        return f"⚠️ لا توجد بيانات كافية للرمز: {ticker}\nجرّب رمز مختلف."

    close = df["Close"]

    # متوسطات
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()

    # اتجاهات
    current_price = float(close.iloc[-1])
    t_short = trend_text(sma20.iloc[-1], sma50.iloc[-1])
    t_medium = trend_text(sma50.iloc[-1], close.rolling(100).mean().iloc[-1])

    # دعم/مقاومة
    sup, res = swing_levels(close, lookback=25)

    # RSI
    rsi = RSIIndicator(close, window=14).rsi().iloc[-1]
    rsi_note = "قوة شرائية" if rsi > 55 else ("تشبع شرائي" if rsi >= 70 else ("قوة بيعية" if rsi < 45 else "تعادل نسبي"))

    # أهداف/وقف خسارة (بسيطة)
    # الهدف = المقاومة الحالية ثم امتداد بسيط؛ الوقف تحت الدعم
    target1 = round(res, 2)
    target2 = round(res * 1.035, 2)
    stop = round(sup * 0.985, 2)

    # نقاط شراء إرشادية (إن كسر مقاومة/على ارتداد من دعم)
    buy_break = round(res * 1.002, 2)
    buy_pullback = round(max(sup, sma20.iloc[-1]), 2)

    # صياغة الرسالة بنفس النسق
    lines = []
    lines.append("📈 *EGY STOCKS BOT*")
    lines.append("")
    lines.append(f"السهم: *{ticker}*")
    lines.append(f"السعر الحالي: *{round(current_price,2)}*")
    lines.append("")
    lines.append(f"الاتجاه الحالي: *{t_short}*")
    lines.append(f"الاتجاه المتوسط: *{t_medium}*")
    lines.append("")
    lines.append("مستويات الدعم والمقاومة:")
    lines.append(f"• الدعم: *{round(sup,2)}*")
    lines.append(f"• المقاومة: *{round(res,2)}*")
    lines.append("")
    lines.append("توصيات التداول:")
    lines.append(f"• شراء *مغامر* قرب: *{buy_pullback}*  (مع مراقبة حجم التداول)")
    lines.append(f"• شراء *تأكيد اختراق* فوق: *{buy_break}*")
    lines.append(f"• الأهداف: *{target1}* ثم *{target2}*")
    lines.append(f"• وقف الخسارة: أسفل *{stop}*")
    lines.append("")
    lines.append("ملاحظة المؤشرات:")
    lines.append(f"• RSI(14): *{round(rsi,1)}* → {rsi_note}")
    lines.append("")
    lines.append(f"_تاريخ التقييم: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC_")

    return "\n".join(lines)


# ===================== Handlers =====================
def start(update, context):
    txt = (
        "أهلاً بيك 👋\n\n"
        "أنا بوت تحليلات سريعة لأسهم البورصة المصرية.\n"
        "استخدم:\n"
        "• /signal COMI.CA  ← يطلع توصية سريعة للسهم (ممكن تكتب الرمز من غير .CA)\n"
        "• /help            ← المساعدة\n\n"
        "مثال: /signal COMI"
    )
    update.message.reply_text(txt)


def help_cmd(update, context):
    update.message.reply_text(
        "المساعدة:\n"
        "اكتب /signal يليه رمز السهم، مثال:\n"
        "/signal ETEL\n"
        "/signal EGAL\n"
        "وتأكد إن الرمز صحيح على Yahoo Finance."
    )


def signal_cmd(update, context):
    if len(context.args) == 0:
        update.message.reply_text("اكتب الرمز بعد الأمر، مثال: /signal COMI")
        return
    symbol = context.args[0]
    try:
        msg = build_signal_message(symbol)
        update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logger.exception("signal error")
        update.message.reply_text(f"حصل خطأ غير متوقع: {e}")


def unknown(update, context):
    update.message.reply_text("مش فاهم الأمر ده. جرّب /help")


# ===================== Main =====================
def main():
    if not TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN غير موجود في المتغيرات البيئية.")

    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help_cmd))
    dp.add_handler(CommandHandler("signal", signal_cmd))

    dp.add_handler(MessageHandler(Filters.command, unknown))

    # شغّل البوت
    updater.start_polling()
    updater.idle()


if __name__ == "__main__":
    main()
