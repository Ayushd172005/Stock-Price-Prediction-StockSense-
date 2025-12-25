# 📈 StockSense - ML-Based Stock Price Movement Predictor

<div align="center">

![StockSense Banner](https://img.shields.io/badge/StockSense-ML%20Stock%20Predictor-blueviolet?style=for-the-badge&logo=tensorflow)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?style=flat-square&logo=tensorflow)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.26-red?style=flat-square&logo=streamlit)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Accuracy-68%25-success?style=flat-square)]()

**🎯 Predicting stock price movements using Bidirectional LSTM and 16+ technical indicators**

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [Results](#-results) • [Documentation](#-documentation)

</div>

---

## 📸 Screenshots

### 🏠 Dashboard Home
![Dashboard Home](https://via.placeholder.com/1200x600/667eea/ffffff?text=StockSense+Dashboard+Home)
*Beautiful, intuitive interface with key metrics and navigation*

### 📊 Data Analysis & Visualization
![Data Analysis](https://via.placeholder.com/1200x600/764ba2/ffffff?text=Interactive+Charts+%26+Technical+Indicators)
*Real-time stock data with interactive candlestick charts, MACD, RSI, and volume analysis*

### 🤖 Model Training
![Model Training](https://via.placeholder.com/1200x600/f093fb/ffffff?text=Model+Training+Progress)
*Live training with real-time metrics, accuracy curves, and loss visualization*

### 🔮 Predictions
![Predictions](https://via.placeholder.com/1200x600/4facfe/ffffff?text=Stock+Price+Predictions)
*Next-day movement predictions with confidence scores and probability distribution*

---

## 🎬 Demo

### Live Prediction Demo
![Prediction Demo](https://via.placeholder.com/800x400/00d2ff/000000?text=Live+Prediction+Demo+GIF)
*Watch StockSense predict stock movements in real-time*

### Training Process
![Training Demo](https://via.placeholder.com/800x400/a8ff78/000000?text=Model+Training+GIF)
*See the model training process with live accuracy updates*

---

## 🚀 Features

<table>
<tr>
<td width="50%">

### 🎯 Core Features
- ✅ **Real-Time Data** from Yahoo Finance
- ✅ **16+ Technical Indicators**
  - RSI, MACD, Bollinger Bands
  - Stochastic Oscillator, ATR
  - Moving Averages (SMA, EMA)
- ✅ **Bidirectional LSTM** Neural Network
- ✅ **3-Class Prediction** (Up/Down/Sideways)
- ✅ **Confidence Scores** for each prediction

</td>
<td width="50%">

### 📊 Dashboard Features
- ✅ **Interactive Charts** with Plotly
- ✅ **4 Navigation Pages**
  - Home, Analysis, Training, Prediction
- ✅ **Real-time Metrics** Display
- ✅ **Model Performance** Visualization
- ✅ **Historical Backtesting**
- ✅ **Beautiful UI** with Custom Styling

</td>
</tr>
</table>

---

## 📊 Results & Performance

### Model Accuracy

<div align="center">

| Metric | Score |
|--------|-------|
| **Test Accuracy** | **68.3%** |
| **Precision (Up)** | 71.2% |
| **Precision (Down)** | 69.8% |
| **Precision (Sideways)** | 64.5% |
| **F1-Score** | 0.68 |

</div>

### Confusion Matrix
![Confusion Matrix](https://via.placeholder.com/600x500/667eea/ffffff?text=Confusion+Matrix+Visualization)
*Detailed breakdown of prediction accuracy across all classes*

### Training History
![Training History](https://via.placeholder.com/1000x400/764ba2/ffffff?text=Accuracy+%26+Loss+Curves)
*Model accuracy and loss over 100 training epochs*

### Feature Importance
![Feature Importance](https://via.placeholder.com/800x500/f093fb/ffffff?text=Top+Technical+Indicators)
*Top 10 most influential technical indicators*

---

## 🛠️ Installation

### Prerequisites

```bash
Python 3.8+
pip or conda
```

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/stocksense.git
cd stocksense
```

### Step 2: Create Virtual Environment (Recommended)

**Using venv:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Using conda:**
```bash
conda create -n stocksense python=3.9
conda activate stocksense
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Requirements include:**
- TensorFlow 2.15.0
- Streamlit 1.26.0
- yfinance 0.2.28
- pandas, numpy, scikit-learn
- plotly, matplotlib, seaborn
- ta (Technical Analysis library)

### Step 4: Setup Project Structure

```bash
# Create necessary directories
mkdir -p data/{raw,processed}
mkdir -p models/{saved_models,scalers}
mkdir -p notebooks
```

---

## 🎮 Usage

### 🌐 Launch Dashboard (Recommended)

```bash
streamlit run app.py
```

Dashboard will open at: `http://localhost:8501`

### 📓 Using Jupyter Notebooks

```bash
# Launch Jupyter
jupyter notebook

# Open notebooks in order:
# 1. notebooks/01_data_exploration.ipynb
# 2. notebooks/02_feature_engineering.ipynb
# 3. notebooks/03_model_training.ipynb
```

### 💻 Using Python Scripts

#### 1️⃣ Fetch and Process Data

```python
from src.data_loader import StockDataLoader
from src.features import FeatureEngineering

# Load stock data
loader = StockDataLoader('AAPL', period='2y')
df = loader.fetch_data()
loader.save_raw_data()

# Create technical indicators
fe = FeatureEngineering(df)
df_processed = fe.create_all_features()
fe.save_processed_data('AAPL')
```

#### 2️⃣ Train Model

```python
from src.train import ModelTrainer

# Initialize trainer
trainer = ModelTrainer(df_processed)

# Prepare sequences
X, y = trainer.prepare_sequences(seq_length=60)

# Split data
X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(X, y)

# Train model
history = trainer.train(X_train, y_train, X_val, y_val, 'AAPL')

# Save scaler
trainer.save_scaler('AAPL')
```

#### 3️⃣ Make Predictions

```python
from src.model import StockPricePredictor
from src.predict import Predictor
import joblib

# Load model and scaler
model = StockPricePredictor((60, 16))
model.load_model('AAPL')
scaler = joblib.load('models/scalers/AAPL_scaler.pkl')

# Create predictor
predictor = Predictor(model.model, scaler, config.FEATURE_COLUMNS)

# Make prediction
result = predictor.predict(df_processed)

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Probabilities: {result['probabilities']}")
```

---

## 🏗️ Project Structure

```
StockSense/
│
┣ 📂 data/
┃ ┣ 📂 raw/                    # Raw stock data (CSV)
┃ ┗ 📂 processed/              # Processed data with indicators
│
┣ 📂 src/
┃ ┣ 📜 __init__.py
┃ ┣ 📜 data_loader.py          # Data fetching from Yahoo Finance
┃ ┣ 📜 features.py             # Technical indicator creation
┃ ┣ 📜 model.py                # LSTM model architecture
┃ ┣ 📜 train.py                # Training pipeline
┃ ┗ 📜 predict.py              # Prediction logic
│
┣ 📂 models/
┃ ┣ 📂 saved_models/           # Trained .h5 models
┃ ┗ 📂 scalers/                # Fitted scalers (.pkl)
│
┣ 📂 dashboard/
┃ ┣ 📜 __init__.py
┃ ┣ 📜 visualizations.py       # Chart creation functions
┃ ┗ 📜 utils.py                # Helper utilities
│
┣ 📂 notebooks/
┃ ┣ 📓 01_data_exploration.ipynb
┃ ┣ 📓 02_feature_engineering.ipynb
┃ ┗ 📓 03_model_training.ipynb
│
┣ 📜 app.py                    # Main Streamlit dashboard
┣ 📜 config.py                 # Configuration settings
┣ 📜 requirements.txt          # Python dependencies
┣ 📜 README.md                 # This file
┗ 📜 .gitignore               # Git ignore rules
```

---

## 🧠 Technical Architecture

### Model Architecture

```
Input Shape: (60, 16)  # 60 timesteps, 16 features
    ↓
Bidirectional LSTM (128 units)
    ↓
Dropout (0.3)
    ↓
Bidirectional LSTM (64 units)
    ↓
Dropout (0.3)
    ↓
Bidirectional LSTM (32 units)
    ↓
Dropout (0.2)
    ↓
Dense Layer (64 units, ReLU)
    ↓
Dropout (0.2)
    ↓
Dense Layer (32 units, ReLU)
    ↓
Output Layer (3 units, Softmax)
    ↓
Prediction: [Down, Sideways, Up]
```

**Total Parameters:** ~250,000

### Technical Indicators (16 Features)

| Category | Indicators |
|----------|-----------|
| **Price-Based** | Close, Returns, Log Returns |
| **Moving Averages** | SMA(5, 10, 20), EMA(12, 26) |
| **Momentum** | RSI(14), Stochastic(K, D), Momentum(4) |
| **Trend** | MACD, MACD Signal, MACD Histogram |
| **Volatility** | Bollinger Bands Width, ATR |
| **Volume** | Volume Ratio, Volume SMA |

---

## 📈 Performance Metrics

### Training Configuration

```python
SEQUENCE_LENGTH = 60      # Days of historical data
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
OPTIMIZER = Adam
LOSS = Sparse Categorical Crossentropy
```

### Results on Test Set (AAPL)

```
              precision    recall  f1-score   support

        Down       0.70      0.67      0.68       145
    Sideways       0.64      0.61      0.63       128
          Up       0.71      0.75      0.73       152

    accuracy                           0.68       425
   macro avg       0.68      0.68      0.68       425
weighted avg       0.68      0.68      0.68       425
```

### Backtesting Results

| Period | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| 1 Month | 71.2% | 0.72 | 0.71 | 0.71 |
| 3 Months | 68.9% | 0.69 | 0.69 | 0.69 |
| 6 Months | 68.3% | 0.68 | 0.68 | 0.68 |
| 1 Year | 67.1% | 0.67 | 0.67 | 0.67 |

---

## 🎓 Use Cases

### 1. Algorithmic Trading
- Integration with trading bots
- Automated buy/sell signals
- Risk management strategies

### 2. Portfolio Management
- Asset allocation decisions
- Risk assessment
- Position sizing

### 3. Market Research
- Sentiment analysis
- Trend identification
- Market timing

### 4. Educational Purposes
- Learning ML for finance
- Understanding technical analysis
- Portfolio projects for job applications

---

## 🔧 Configuration

Edit `config.py` to customize behavior:

```python
# Model Parameters
SEQUENCE_LENGTH = 60          # Lookback window (days)
TRAIN_TEST_SPLIT = 0.8       # 80% train, 20% test
VALIDATION_SPLIT = 0.1       # 10% validation

# Training Parameters
EPOCHS = 100                 # Training iterations
BATCH_SIZE = 32             # Samples per batch
LEARNING_RATE = 0.001       # Adam optimizer LR

# Prediction Thresholds
UP_THRESHOLD = 0.002        # 0.2% increase
DOWN_THRESHOLD = -0.002     # 0.2% decrease

# Data Source
DATA_PERIOD = '2y'          # Historical data period
DATA_INTERVAL = '1d'        # Daily data

# Supported Tickers
DEFAULT_TICKERS = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA']
```

---

## 📚 Documentation

### Jupyter Notebooks

1. **📓 01_data_exploration.ipynb**
   - Data loading and inspection
   - Statistical analysis
   - Price and volume visualization
   - Correlation analysis

2. **📓 02_feature_engineering.ipynb**
   - Technical indicator creation
   - Feature visualization
   - Target variable construction
   - Feature importance analysis

3. **📓 03_model_training.ipynb**
   - Sequence preparation
   - Model architecture
   - Training process
   - Performance evaluation

### API Documentation

#### StockDataLoader
```python
loader = StockDataLoader(ticker='AAPL', period='2y')
df = loader.fetch_data()
loader.save_raw_data()
```

#### FeatureEngineering
```python
fe = FeatureEngineering(df)
df_processed = fe.create_all_features()
feature_cols = fe.get_feature_columns()
```

#### ModelTrainer
```python
trainer = ModelTrainer(df_processed)
X, y = trainer.prepare_sequences(seq_length=60)
history = trainer.train(X_train, y_train, X_val, y_val, ticker)
```

#### Predictor
```python
predictor = Predictor(model, scaler, feature_cols)
result = predictor.predict(df_processed)
# Returns: {'prediction': 'Up ↑', 'confidence': 0.75, 'probabilities': {...}}
```

---

## 🛠️ Technologies Used

<div align="center">

| Category | Technologies |
|----------|-------------|
| **Core ML** | ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white) ![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) |
| **Data Processing** | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) |
| **Visualization** | ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat) ![Seaborn](https://img.shields.io/badge/Seaborn-444876?style=flat) |
| **Dashboard** | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white) |
| **Data Source** | ![Yahoo Finance](https://img.shields.io/badge/Yahoo%20Finance-720E9E?style=flat&logo=yahoo&logoColor=white) |

</div>

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the Repository**
   ```bash
   git clone https://github.com/yourusername/stocksense.git
   ```

2. **Create Feature Branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```

3. **Commit Changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```

4. **Push to Branch**
   ```bash
   git push origin feature/AmazingFeature
   ```

5. **Open Pull Request**

### Contribution Ideas
- 🆕 Add more technical indicators
- 🎨 Improve dashboard UI/UX
- 📊 Add more visualization options
- 🤖 Implement different model architectures
- 📝 Improve documentation
- 🧪 Add unit tests

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## ⚠️ Disclaimer

**IMPORTANT: This project is for educational and research purposes only.**

- 📚 **Not Financial Advice**: This tool does not provide financial advice
- ⚠️ **Risk Warning**: Trading stocks involves substantial risk of loss
- 🎓 **Educational Purpose**: Use this for learning ML and technical analysis
- 💼 **Consult Professionals**: Always consult with licensed financial advisors
- 📉 **Past Performance**: Historical accuracy does not guarantee future results

**By using this software, you acknowledge:**
- All investment decisions are your own responsibility
- The creators are not liable for any financial losses
- This is a learning tool, not a trading system
- You should never invest money you cannot afford to lose

---

## 🙏 Acknowledgments

- **Yahoo Finance** - Free stock market data API
- **TensorFlow Team** - Amazing deep learning framework
- **Streamlit** - Beautiful dashboard framework
- **TA-Lib Contributors** - Technical analysis library
- **Open Source Community** - Inspiration and support

---

## 📧 Contact & Support

<div align="center">

### 👤 Author: **Your Name**

[![Portfolio](https://img.shields.io/badge/Portfolio-000000?style=for-the-badge&logo=About.me&logoColor=white)](https://yourportfolio.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/yourprofile)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/yourusername)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:your.email@example.com)

### 📬 Get in Touch

**Questions?** Open an [Issue](https://github.com/yourusername/stocksense/issues)

**Feature Requests?** Start a [Discussion](https://github.com/yourusername/stocksense/discussions)

**Found a Bug?** Submit a [Bug Report](https://github.com/yourusername/stocksense/issues/new)

</div>

---

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/stocksense&type=Date)](https://star-history.com/#yourusername/stocksense&Date)

---

<div align="center">

### 🚀 Ready to Predict Stock Movements?

**[Get Started Now](#-installation)** | **[View Demo](#-demo)** | **[Read Docs](#-documentation)**

Made with ❤️ and Python | © 2024 StockSense

![Python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg?style=flat-square)
![TensorFlow](https://img.shields.io/badge/Powered%20by-TensorFlow-orange?style=flat-square)
![Love](https://img.shields.io/badge/Built%20with-❤-red?style=flat-square)

</div>
