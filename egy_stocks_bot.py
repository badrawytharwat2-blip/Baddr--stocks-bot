
# -*- coding: utf-8 -*-
# EGY Stocks Signal Bot (Arabic UI)
# ---------------------------------
# متطلبات التشغيل:
# - Python 3.10+
# - مكتبات: python-telegram-bot==20.*, pandas, numpy, yfinance, ta, python-dotenv (اختياري)
# - متغير بيئة: TELEGRAM_BOT_TOKEN
#
# ملاحظات هامة:
# - بيانات EGX على Yahoo قد تتطلب رمزًا منتهيًا بـ ".CA" مثل "COMI.CA" (CIB).
# - يمكنك استبدال مزود البيانات (DataProvider) بمزود آخر يناسب EGX إن أردت.

import os
import logging
from datetime import timedelta

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:
    yf = None

try:
    import ta
except Exception:
    ta = None

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler,
    ContextTypes
)

# --------------------------- إعدادات عامة ---------------------------
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("EGYStocksBot")

BOT_NAME = "EGY STOCKS BOT"
DEFAULT_PERIOD = "6mo"      # مدة البيانات
DEFAULT_INTERVAL = "1d"     # الفاصل الزمني

# --------------------------- مزود البيانات ---------------------------
class DataProvider:
    """واجهة عامة لمزود البيانات."""
    def fetch(self, symbol: str, period: str = DEFAULT_PERIOD, interval: str = DEFAULT_INTERVAL) -> pd.DataFrame:
        raise NotImplementedError

class YahooProvider(DataProvider):
    def fetch(self, symbol: str, period: str = DEFAULT_PERIOD, interval: str = DEFAULT_INTERVAL) -> pd.DataFrame:
        if yf is None:
            raise RuntimeError("مكتبة yfinance غير مثبتة.")
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval, auto_adjust=False)
        if df is None or df.empty:
            raise ValueError("تعذر جلب بيانات الرمز. تأكد من الرمز مثل COMI.CA أو ORHD.CA ...")
        df = df.rename(columns=str.title)  # Open, High, Low, Close, Volume
        df = df.dropna()
        return df

DATA_PROVIDER: DataProvider = YahooProvider()

# --------------------------- التحليل الفني ---------------------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """يضيف أعمدة المؤشرات الفنية اللازمة."""
    if ta is None:
        raise RuntimeError("مكتبة ta غير مثبتة. ثبّت 'ta' عبر pip.")
    out = df.copy()
    out["EMA_20"] = ta.trend.EMAIndicator(out["Close"], window=20).ema_indicator()
    out["EMA_50"] = ta.trend.EMAIndicator(out["Close"], window=50).ema_indicator()
    out["EMA_200"] = ta.trend.EMAIndicator(out["Close"], window=200).ema_indicator()
    out["RSI_14"] = ta.momentum.RSIIndicator(out["Close"], window=14).rsi()
    out["ATR_14"] = ta.volatility.AverageTrueRange(out["High"], out["Low"], out["Close"], window=14).average_true_range()
    return out.dropna()

def swing_levels(series: pd.Series, window: int = 10, lookback: int = 60):
    """أقرب دعوم ومقاومات بسيطة من آخر 60 يومًا عبر قيعان/قمم محلية."""
    s = series.tail(lookback).reset_index(drop=True)
    lows = []
    highs = []
    for i in range(window, len(s) - window):
        chunk = s.iloc[i-window:i+window+1]
        val = s.iloc[i]
        if val == chunk.min():
            lows.append(float(val))
        if val == chunk.max():
            highs.append(float(val))
    lows = sorted(list(set([round(x, 2) for x in lows])))
    highs = sorted(list(set([round(x, 2) for x in highs])))
    # نعيد آخر 5 دعوم (أقرب للأسعار الحديثة) وأول 5 مقاومات
    return lows[-5:], highs[:5]

def trend_label(current_price: float, ema_short: float, ema_mid: float, ema_long: float):
    now_trend = "صاعد" if current_price >= ema_short else "هابط"
    mid_trend = "صاعد" if ema_mid >= ema_long else "هابط"
    return now_trend, mid_trend

def generate_reco_text(df: pd.DataFrame, symbol: str) -> str:
    row = df.iloc[-1]
    price = round(row["Close"], 2)
    ema20 = round(row["EMA_20"], 2)
    ema50 = round(row["EMA_50"], 2)
    ema200 = round(row["EMA_200"], 2)
    rsi = round(row["RSI_14"], 1)
    atr = float(row["ATR_14"])

    now_trend, mid_trend = trend_label(price, row["EMA_20"], row["EMA_50"], row["EMA_200"])

    # دعوم ومقاومات
    lows, highs = swing_levels(df["Close"])
    supports = ", ".join([f"{x:.2f}" for x in lows]) if lows else "—"
    resistances = ", ".join([f"{x:.2f}" for x in highs]) if highs else "—"

    # نقاط شراء/وقف/أهداف مبسطة
    last_20 = df["Close"].tail(20)
    swing_low = float(last_20.min())
    swing_high = float(last_20.max())
    fib_382 = swing_high - 0.382 * (swing_high - swing_low)
    buy_zone = round((row["EMA_20"] + fib_382) / 2, 2)

    stop_base = lows[-1] if lows else price * 0.93
    stop = round(stop_base - 0.5 * atr, 2)

    if highs:
        # اختر أول مستوى أعلى من السعر إن وجد، وإلا أول مستوى على أي حال
        higher_res = [h for h in highs if h > price]
        t1 = higher_res[0] if higher_res else highs[0]
        t2 = highs[1] if len(highs) > 1 else price + 1.5 * atr
    else:
        t1 = price + 1.0 * atr
        t2 = price + 2.0 * atr
    t1, t2 = round(float(t1), 2), round(float(t2), 2)

    if price > row["EMA_20"] > row["EMA_50"] and rsi < 70:
        idea = f"شراء مضارِب حول {buy_zone} بهدف {t1} ثم {t2} ووقف أسفل {stop}."
    elif row["EMA_20"] > price > row["EMA_50"] and rsi <= 50:
        idea = f"انتظار هبوط لمنطقة {buy_zone} ثم متابعة اختراق {t1}."
    elif price < row["EMA_50"] and rsi < 40:
        idea = "ضعيف حاليًا؛ يفضّل المراقبة حتى استعادة السعر فوق EMA50."
    else:
        idea = "محايد؛ ننتظر إشارة أوضح (اختراق/كسر)."

    # صياغة الرسالة
    lines = []
    lines.append(f"*{BOT_NAME}*")
    lines.append(f"الرمز: *{symbol}* — السعر: *{price}*")
    lines.append("")
    lines.append(f"الاتجاه الحالي: *{now_trend}*")
    lines.append(f"الاتجاه المتوسط: *{mid_trend}*")
    lines.append("")
    lines.append("مستويات الدعم:")
    lines.append(f"`{supports}`")
    lines.append("")
    lines.append("مستويات المقاومة:")
    lines.append(f"`{resistances}`")
    lines.append("")
    lines.append("توصيات التداول:")
    lines.append(f"{idea}")
    lines.append("")
    lines.append(f"نقاط الشراء الرئيسية: *{buy_zone}*")
    lines.append(f"المستهدفات: *{t1}* ثم *{t2}*  | وقف الخسارة: *{stop}*")
    lines.append("")
    lines.append(f"_مؤشرات:_ EMA20={ema20} | EMA50={ema50} | EMA200={ema200} | RSI14={rsi}")
    return "\n".join(lines)

# --------------------------- أوامر البوت ---------------------------
def example_symbols_text() -> str:
    return "أمثلة لرموز EGX على ياهو: COMI.CA (CIB), EFIH.CA, SWDY.CA, TMGH.CA, ORHD.CA"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = [
        [InlineKeyboardButton("إشارة/Signal", switch_inline_query_current_chat="signal COMI.CA")],
        [InlineKeyboardButton("مساعدة", callback_data="help")]
    ]
    await update.message.reply_text(
        f"مرحبًا! أنا {BOT_NAME}.\nأرسل الأمر: /signal <الرمز> مثل /signal COMI.CA\n{example_symbols_text()}",
        reply_markup=InlineKeyboardMarkup(kb)
    )

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "الأوامر المتاحة:\n"
        "/signal <رمز>  — إشارة تحليل فني مختصرة بنفس نسق الصورة.\n"
        "/watchlist add <رمز> — إضافة للمتابعة\n"
        "/watchlist remove <رمز> — حذف من المتابعة\n"
        "/watchlist show — عرض قائمة المتابعة\n"
        "مثال: /signal COMI.CA"
    )
    await update.message.reply_text(text)

def get_user_watchlist(context: ContextTypes.DEFAULT_TYPE):
    return context.user_data.setdefault("watchlist", set())

async def watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    wl = get_user_watchlist(context)
    if not args or args[0] == "show":
        if not wl:
            await update.message.reply_text("قائمة المتابعة فارغة.")
        else:
            await update.message.reply_text("قائمة المتابعة:\n" + ", ".join(sorted(wl)))
        return
    if args[0] == "add" and len(args) >= 2:
        wl.add(args[1].upper())
        await update.message.reply_text(f"تمت إضافة {args[1].upper()}")
    elif args[0] == "remove" and len(args) >= 2:
        wl.discard(args[1].upper())
        await update.message.reply_text(f"تم حذف {args[1].upper()}")
    else:
        await update.message.reply_text("استخدام: /watchlist add <رمز> | remove <رمز> | show")

async def signal_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("استخدم: /signal <رمز> مثل /signal COMI.CA")
        return
    symbol = context.args[0].upper()
    try:
        df_raw = DATA_PROVIDER.fetch(symbol)
        df = compute_indicators(df_raw)
        text = generate_reco_text(df, symbol)
        await update.message.reply_markdown(text)
    except Exception as e:
        logger.exception(e)
        await update.message.reply_text(f"تعذر إصدار الإشارة: {e}\n{example_symbols_text()}")

async def inline_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if not query:
        return
    if query.data == "help":
        await query.answer()
        await query.edit_message_text(
            "أرسل /signal <رمز> للحصول على إشارة. يمكنك أيضًا استخدام /watchlist لإدارة قائمة المتابعة."
        )

def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("يرجى ضبط متغير البيئة TELEGRAM_BOT_TOKEN")
    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("signal", signal_cmd))
    app.add_handler(CommandHandler("watchlist", watchlist))
    app.add_handler(CallbackQueryHandler(inline_help))

    logger.info("Starting bot...")
    app.run_polling()

if __name__ == "__main__":
    main()
