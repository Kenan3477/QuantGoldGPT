
@app.route('/api/ml-predictions/daily')
def get_daily_ml_predictions():
    """Get today's ML predictions only (max 6 predictions)"""
    try:
        conn = sqlite3.connect('goldgpt_ml_tracking.db')
        cursor = conn.cursor()
        
        # Get only today's predictions
        cursor.execute("""
            SELECT engine_name, timeframe, predicted_price, current_price,
                   change_percent, direction, confidence, created_at
            FROM ml_engine_predictions 
            WHERE DATE(created_at) = DATE('now', 'localtime')
            ORDER BY created_at DESC
        """)
        
        predictions = cursor.fetchall()
        conn.close()
        
        # Format results
        daily_predictions = {
            'enhanced_ml_prediction_engine': [],
            'intelligent_ml_predictor': []
        }
        
        for pred in predictions:
            engine_name, timeframe, predicted_price, current_price, change_percent, direction, confidence, created_at = pred
            
            prediction_data = {
                'timeframe': timeframe,
                'predicted_price': predicted_price,
                'current_price': current_price,
                'change_percent': change_percent,
                'direction': direction,
                'confidence': confidence,
                'timestamp': created_at
            }
            
            if engine_name in daily_predictions:
                daily_predictions[engine_name].append(prediction_data)
        
        return jsonify({
            'success': True,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'total_predictions': len(predictions),
            'predictions': daily_predictions,
            'message': f"Today's predictions only - max 6 total",
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500
