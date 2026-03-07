import os
import sys
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.loader import load_dataset
from src.data.features import createFeatures
from src.models.predictor import make_prediction
from src.models.trainer import train_model

load_dotenv()

TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TOKEN:
    raise ValueError("No token in env")

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "model.pkl")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hello, I'm a trading bot\n\n"
        "Commands:\n"
        "/predict AAPL - prediction for stocks"
    )

async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("⚠️ Name ticker. Example: /predict AAPL")
        return
    
    ticker = context.args[0].upper()
    await update.message.reply_text(f"🔄 Loading data for {ticker}...")

    try:
        df = load_dataset(ticker)
        df_features = createFeatures(df)
        result = make_prediction(df_features, model_path=MODEL_PATH)

        message = (
            f"📊 {ticker}\n"
            f"📅 Date: {result['date']}\n"
            f"💵 Price: ${result['price']}\n"
            f"🎯 Signal: {result['signal']}\n"
            f"📈 Confidence: {result['confidence']}"
        )

        await update.message.reply_text(message)
    
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")

def main():
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("predict", predict))
    print("Бот запущен...")
    app.run_polling()

if __name__ == "__main__":
    main()
