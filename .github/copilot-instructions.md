<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# GoldGPT - Advanced AI Trading Web Application

## Project Overview
GoldGPT is a sophisticated web-based trading platform inspired by Trading 212's design and functionality. It adapts advanced AI trading capabilities from a Telegram bot to a modern web interface.

## Key Features
- Real-time trading dashboard with Trading 212-inspired UI
- Advanced AI analysis including technical, sentiment, and ML predictions
- Real-time price updates via WebSocket
- Portfolio management and trade execution
- Comprehensive market analysis tools

## Technology Stack
- **Backend**: Flask, Flask-SocketIO, SQLite
- **Frontend**: HTML5, CSS3, JavaScript (ES6+), Socket.IO
- **AI/ML**: NumPy, Pandas, Scikit-learn, TensorFlow
- **Real-time**: WebSocket for live updates
- **Database**: SQLite (development), easily upgradable to PostgreSQL

## Code Style Guidelines
- Use Python PEP 8 style for backend code
- Use modern JavaScript ES6+ features
- Follow Flask best practices for route organization
- Use semantic HTML and modern CSS grid/flexbox
- Implement responsive design patterns

## Architecture Patterns
- MVC pattern for Flask application structure
- Modular design with separate advanced_systems module
- Event-driven architecture with WebSocket integration
- RESTful API design for frontend-backend communication

## AI/ML Integration
- Modular AI systems for technical analysis, sentiment analysis, and ML predictions
- Ensemble methods for robust predictions
- Real-time data processing and analysis
- Confidence scoring for all AI recommendations

## Security Considerations
- Environment variables for sensitive configuration
- Input validation for all API endpoints
- Rate limiting for trading operations
- Secure WebSocket connections

## Development Guidelines
- Write comprehensive docstrings for all functions
- Include error handling and logging
- Use type hints where appropriate
- Follow Trading 212's design principles for UI consistency
- Implement graceful degradation for advanced features

## Testing Strategy
- Unit tests for all AI analysis functions
- Integration tests for API endpoints
- WebSocket connection testing
- Portfolio calculation accuracy tests

## Deployment Considerations
- Environment-specific configuration
- Database migration support
- Real-time feature scaling
- Performance monitoring integration
