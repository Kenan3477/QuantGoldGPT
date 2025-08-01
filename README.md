# 🏆 GoldGPT - Advanced AI Trading Web Application

<div align="center">

![GoldGPT Logo](https://img.shields.io/badge/GoldGPT-Advanced%20AI%20Trading-gold?style=for-the-badge&logo=chart-line)

A sophisticated Trading 212-inspired web platform featuring advanced AI trading capabilities, real-time market analysis, and comprehensive portfolio management.

[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green?style=flat-square&logo=flask)](https://flask.palletsprojects.com)
[![WebSocket](https://img.shields.io/badge/WebSocket-Real--time-orange?style=flat-square)](https://socket.io)
[![AI/ML](https://img.shields.io/badge/AI%2FML-Advanced-purple?style=flat-square&logo=tensorflow)](https://tensorflow.org)

</div>

## 🚀 Features

### 📊 Trading Dashboard
- **Trading 212-inspired interface** with modern, responsive design
- **Real-time price updates** via WebSocket connections
- **Interactive charts** and technical analysis tools
- **Portfolio overview** with live P&L calculations

### 🤖 Advanced AI Analysis
- **Technical Analysis**: RSI, MACD, SMA, EMA, Bollinger Bands
- **Sentiment Analysis**: Multi-source market sentiment processing
- **ML Predictions**: Ensemble models with confidence scoring
- **Pattern Detection**: Chart pattern recognition system
- **Macro Analysis**: Economic indicators and market correlations

### 💼 Portfolio Management
- **Real-time trade execution** with instant confirmations
- **Position management** with stop-loss and take-profit
- **Trade history** and performance analytics
- **Risk management** tools and position sizing

### 🔄 Real-time Features
- **Live price feeds** for major currency pairs and commodities
- **Instant notifications** for trade executions and market events
- **WebSocket integration** for seamless real-time updates
- **Multi-symbol monitoring** with customizable watchlists

## 🛠️ Technology Stack

### Backend
- **Flask** - Web framework with RESTful API design
- **Flask-SocketIO** - Real-time WebSocket communication
- **SQLite** - Database (easily upgradable to PostgreSQL)
- **NumPy/Pandas** - Data processing and analysis
- **Scikit-learn/TensorFlow** - Machine learning models

### Frontend
- **HTML5/CSS3** - Modern, responsive interface
- **JavaScript ES6+** - Interactive user experience
- **Socket.IO** - Real-time client-server communication
- **Chart.js** - Interactive financial charts

### AI/ML Systems
- **Technical Analysis Engine** - Multi-indicator analysis
- **Sentiment Analysis** - News and social media processing
- **Machine Learning Models** - Predictive analytics
- **Pattern Recognition** - Chart pattern detection

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (optional)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd GoldGPT
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   .\venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment setup**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Initialize database**
   ```bash
   python -c "from app import init_database; init_database()"
   ```

6. **Run the application**
   ```bash
   python app.py
   ```

7. **Access the dashboard**
   Open your browser and navigate to `http://localhost:5000`

## ⚙️ Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```env
# Flask Configuration
SECRET_KEY=your-secret-key-here
FLASK_ENV=development

# API Keys (optional for real data)
ALPHA_VANTAGE_API_KEY=your-key
FINNHUB_API_KEY=your-key

# Trading Settings
ENABLE_REAL_TRADING=False
ENABLE_ADVANCED_AI=True

# Database
DATABASE_URL=sqlite:///goldgpt.db
```

### Feature Flags
- `ENABLE_REAL_TRADING`: Enable actual trade execution (default: False)
- `ENABLE_ADVANCED_AI`: Enable AI analysis features (default: True)
- `ENABLE_NOTIFICATIONS`: Enable real-time notifications (default: True)

## 🎯 Usage

### Dashboard Navigation
- **📊 Dashboard**: Main overview with charts and portfolio
- **💼 Portfolio**: Detailed position management
- **📈 Trading**: Order placement and execution
- **🤖 AI Analysis**: Advanced market analysis tools
- **📋 History**: Trade history and performance
- **⚙️ Settings**: Application configuration

### Trading Operations
1. **View Market Data**: Real-time prices and charts
2. **Run AI Analysis**: Technical, sentiment, and ML predictions
3. **Execute Trades**: Buy/sell with automatic position tracking
4. **Manage Portfolio**: Monitor positions and P&L
5. **Analyze Performance**: Historical trade analysis

### AI Analysis
- **Technical Analysis**: Click "Technical Analysis" for indicator-based insights
- **Sentiment Analysis**: Market sentiment from multiple sources
- **ML Predictions**: Ensemble model predictions with confidence scores
- **Comprehensive Analysis**: Combined analysis with actionable recommendations

## 🔧 Development

### Project Structure
```
GoldGPT/
├── app.py                 # Main Flask application
├── advanced_systems.py    # AI analysis modules
├── config.py             # Configuration management
├── requirements.txt      # Python dependencies
├── templates/
│   └── dashboard.html    # Main dashboard template
├── static/
│   └── js/
│       └── app.js       # Frontend JavaScript
└── .github/
    └── copilot-instructions.md
```

### Adding New Features
1. **Backend**: Add new routes in `app.py`
2. **AI Systems**: Extend `advanced_systems.py`
3. **Frontend**: Update `templates/dashboard.html` and `static/js/app.js`
4. **Database**: Modify schema in `init_database()` function

### API Endpoints
- `GET /api/portfolio` - Get portfolio data
- `GET /api/analysis/<symbol>` - Get AI analysis
- `POST /api/trade` - Execute trade
- `POST /api/close_trade/<id>` - Close position

## 🤖 AI Systems

### Technical Analysis
- **Indicators**: RSI, MACD, SMA, EMA, Bollinger Bands
- **Signals**: Buy/Sell/Hold recommendations
- **Support/Resistance**: Dynamic level calculation

### Sentiment Analysis
- **Sources**: News, social media, economic data
- **Scoring**: -1 (bearish) to +1 (bullish)
- **Confidence**: Statistical confidence levels

### Machine Learning
- **Models**: LSTM, Random Forest, SVM, XGBoost
- **Ensemble**: Combined predictions for higher accuracy
- **Confidence**: Model agreement and prediction strength

## 📊 Performance

### Optimization Features
- **Real-time Updates**: WebSocket for minimal latency
- **Efficient Database**: Optimized SQLite queries
- **Caching**: Price data and analysis caching
- **Lazy Loading**: On-demand module loading

### Scalability
- **Modular Architecture**: Easy to extend and modify
- **Database Agnostic**: Easy migration to PostgreSQL
- **Horizontal Scaling**: WebSocket clustering support
- **API Rate Limiting**: Configurable request limits

## 🔒 Security

### Security Features
- **Environment Variables**: Sensitive data protection
- **Input Validation**: All API endpoints protected
- **Rate Limiting**: Trading operation limits
- **Secure Connections**: HTTPS and WSS support

### Best Practices
- Regular security updates
- API key rotation
- Database encryption
- Audit logging

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Guidelines
- Follow PEP 8 for Python code
- Use modern JavaScript ES6+ features
- Write comprehensive docstrings
- Include error handling
- Update documentation

## 📄 License

This project is proprietary software owned by Kenan Davies. All rights reserved.

**Copyright Notice**: This software contains proprietary algorithms and trading strategies. Commercial use, reverse engineering, or derivative works are strictly prohibited without written permission.

## 🆘 Support

### Troubleshooting
- **Installation Issues**: Check Python version and dependencies
- **WebSocket Errors**: Verify firewall and network settings
- **Database Errors**: Ensure proper SQLite installation
- **API Errors**: Check configuration and API keys

### Common Issues
1. **Port 5000 in use**: Change port in `app.py`
2. **Module import errors**: Verify virtual environment activation
3. **WebSocket connection failed**: Check browser WebSocket support
4. **Analysis not working**: Verify advanced systems configuration

---

<div align="center">

**Built with ❤️ for advanced trading**

*GoldGPT - Where AI meets Trading Excellence*

</div>
#   N a v i g a t i o n   f i x   d e p l o y e d   2 0 2 5 - 0 8 - 0 1   2 0 : 4 2 : 0 2  
 