# ai-trading-telegram-bot

AI-powered Telegram bot that predicts stock price direction using Machine Learning.

## What it does
- Loads historical stock data from Yahoo Finance
- Builds technical indicators (SMA, RSI, MACD, Momentum)
- Trains a Logistic Regression model to predict UP/DOWN
- Returns prediction via Telegram bot

## Project Structure
```
ai-trading-telegram-bot/
├── src/
│   ├── data/
│   │   ├── loader.py       # Load raw OHLCV data
│   │   └── features.py     # Feature engineering + target
│   ├── models/
│   │   ├── trainer.py      # Model training + saving
│   │   └── predictor.py    # Load model + inference
│   └── bot/
│       └── bot.py          # Telegram bot
├── models/
│   └── model.pkl           # Saved model + scaler
├── notebooks/
│   └── 01_data_preview.ipynb
├── .env                    # Telegram token (never commit)
├── .gitignore
└── requirements.txt
```

## Setup
```bash
# Clone repo
git clone https://github.com/yourusername/ai-trading-telegram-bot.git
cd ai-trading-telegram-bot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
echo "TELEGRAM_TOKEN=your_token_here" > .env
```

## Train the model
```bash
# Run in notebooks/01_data_preview.ipynb
# or create a train script:
python -c "
import sys; sys.path.append('.')
from src.data.loader import load_raw_data
from src.data.features import build_features
from src.models.trainer import train_model
df = load_raw_data('AAPL')
df = build_features(df)
train_model(df)
"
```

## Run the bot
```bash
source venv/bin/activate
python src/bot/bot.py
```

## Usage
```
/start        — welcome message
/predict AAPL — get prediction for a stock
/help         — list of commands
```

## Tech Stack

- Python 3.12
- python-telegram-bot
- scikit-learn
- yfinance
- pandas

## Architecture

Training and inference are fully separated:
- `trainer.py` runs manually to update the model
- `predictor.py` loads the saved model for every bot request
- The bot knows nothing about ML — it only calls the predictor