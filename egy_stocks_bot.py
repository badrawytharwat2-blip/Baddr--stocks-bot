# -*- coding: utf-8 -*-
import os
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from ta.momentum import RSIIndicator

from telegram import ParseMode
from telegram.ext import Updater, CommandHandler

# ========= إعداد اللوجز =========
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("EGY-STOCKS-BOT")

# ========= مساعدات =========
def egx_ticker(text: str) -> str:
    """
    يضيف .CA لو المستخدم كتب الرمز بدون اللاحقة
    """
    t = (text or "").strip().upper()
    if not t.endswith(".CA"):
        t = f"{t}.CA"
    return t

def fetch_ohlc(ticker: str, period="9mo", interval="1d") -> pd.DataFrame:
    """
    تحميل بيانات الأسعار من ياهو فاينانس
    """
    df = yf.download(ticker, period=period, interval=interval,
                     auto_adjust=True, progress=False)
    if isinstance(df, pd.DataFrame) and not df.empty:
        return df.dropna().copy()
    return None

def swing_levels(series: pd.Series, lookback: int = 25):
    """
    دعم/مقاومة مبسطة: أقل/أعلى قيمة في نافذة حديثة
    """
    recent = series[-lookback:]
    return float(recent.min()), float(recent.max())

def trend_text(sma_fast: float, sma_slow: float) -> str:
    if np.isnan(sma_fast) or np.isnan(sma_slow):
        return "غير واضح"
    return "صاعد" if sma_fast >= sma_slow else "هابط"

def build_signal_message(ticker_raw: str) -> str:
    ticker = egx_ticker(ticker_raw)
    df = fetch_ohlc(ticker, period="9mo", interval="1d")
    if df is None or df.empty or len(df) < 30:
        return f"⚠️ لا توجد بيانات كافية للرمز: {ticker}\nجرّب رمز مختلف."

    close = df["Close"]
    # متوسطات
    sma20  = close.rolling(20).mean()
    sma50  = close.rolling(50).mean()
    sma100 = close.rolling(100).mean()

    current_price = float(close.iloc[-1])

    t_short  = trend_text(sma20.iloc[-1],  sma50.iloc[-1])
    t_medium = trend_text(sma50.iloc[-1],  sma100.iloc[-1])

    # دعم/مقاومة
    sup, res = swing_levels(close, lookback=25)

    # RSI
    rsi = float(RSIIndicator(close, window=14).rsi().iloc[-1])
    rsi_note = (
        "قوّة شرائية"   if rsi >= 70 else
        "تشبّع شرائي"  if rsi > 55  else
        "تعادل نسبي"   if rsi >= 45 else
        "تشبّع بيعي"
    )

    # نقاط إرشادية مبسطة
    target1 = round(res, 2)
    target2 = round(res * 1.035, 2)
    stop    = round(sup * 0.985, 2)
    buy_break     = round(res * 1.002, 2)             # شراء اختراق فوق المقاومة
    buy_pullback  = round(max(sup, sma20.iloc[-1]), 2) # شراء على ارتداد من دعم/متوسط

    lines = []
    lines.append(f"📈 *EGY STOCKS BOT*")
    lines.append("")
    lines.append(f"• السهم: *{ticker}*")
    lines.append(f"• السعر الحالي: *{round(current_price,2)}*")
    lines.append("")
    lines.append(f"الاتجاه الحالي: *{t_short}*")
    lines.append(f"الاتجاه المتوسط: *{t_medium}*")
    lines.append("")
    lines.append("مستويات الدعم والمقاومة:")
    lines.append(f"• الدعم: *{round(sup,2)}*")
    lines.append(f"• المقاومة: *{round(res,2)}*")
    lines.append("")
    lines.append(f"RSI (14): *{round(rsi,1)}* → {rsi_note}")
    lines.append("")
    lines.append("توصيات تداول (إرشادية وليست نصيحة استثمار):")
    lines.append(f"• شراء تأكيدي (اختراق): فوق *{buy_break}*")
    lines.append(f"• شراء ارتداد: قرب *{buy_pullback}* (مع متابعة الحجم)")
    lines.append(f"• الأهداف: *{target1}* ثم *{target2}*")
    lines.append(f"• وقف الخسارة: أسفل *{stop}*")
    lines.append("")
    lines.append(
        "_⚠️ هذا البوت لأغراض تعليمية؛ التداول على مسؤوليتك الشخصية._"
    )
    return "\n".join(lines)

# ========= الأوامر =========
def start(update, context):
    txt = (
        "أهلاً بيك 👋\n\n"
        "أنا بوت تحليلات سريعة لأسهم البورصة المصرية.\n"
        "استخدم:\n"
        "• /signal COMI  ← (اكتب رمز السهم من غير .CA)\n"
        "• /help  ← للمساعدة\n\n"
        "مثال: /signal COMI"
    )
    update.message.reply_text(txt)

def help_cmd(update, context):
    update.message.reply_text(
        "المساعدة:\n"
        "اكتب /signal يليه رمز السهم، مثال:\n"
        "/signal ETEL\n"
        "/signal EGAL\n"
        "/signal SWDY\n\n"
        "الرموز تُجلب من Yahoo Finance (لاحقة .CA تُضاف تلقائياً)."
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
        update.message.reply_text(f"حدث خطأ غير متوقع: {e}")

# ========= التشغيل =========
def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Environment variable TELEGRAM_BOT_TOKEN غير موجود.")

    updater = Updater(token=token, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help_cmd))
    dp.add_handler(CommandHandler("signal", signal_cmd))

    logger.info("Bot is starting…")
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
