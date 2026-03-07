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

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

