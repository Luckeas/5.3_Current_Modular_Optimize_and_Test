"""
backtester_common.py - Common functions for backtesters

This module provides shared functions used by both the standard and ML-enhanced
backtesters, eliminating code duplication.
"""

import numpy as np
import logging
import os
from datetime import datetime, timedelta

# Import the centralized configuration
from config import config

# Import HMM regime detector
from hmm_regime_detector import HMMRegimeDetector

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global HMM detector instance (shared by both backtesters)
hmm_detector = None

#############################################################################
#                    MARKET REGIME FUNCTIONS                                #
#############################################################################

def initialize_hmm_detector(output_dir=None):
    """Initialize the HMM regime detector"""
    global hmm_detector
    
    if output_dir:
        hmm_output_dir = os.path.join(output_dir, 'hmm_detector')
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hmm_output_dir = f"hmm_detector_{timestamp}"
    
    if not os.path.exists(hmm_output_dir):
        os.makedirs(hmm_output_dir)
    
    # Create detector with config settings
    detector_settings = config['hmm_detector']
    hmm_detector = HMMRegimeDetector(
        n_states=detector_settings['n_states'],
        lookback_days=detector_settings['lookback_days'],
        retrain_frequency=detector_settings['retrain_frequency'],
        min_samples=detector_settings['min_samples'],
        output_dir=hmm_output_dir
    )
    
    logger.info(f"Initialized HMM regime detector with output dir: {hmm_output_dir}")
    return hmm_detector


def reset_hmm_detector():
    """
    Reset the HMM detector state completely between runs.
    This ensures that each backtest starts with a fresh HMM state.
    """
    global hmm_detector
    hmm_detector = None

    # Also clear any cached attributes in the detect_market_type function
    if hasattr(detect_market_type, 'historical_scores'):
        delattr(detect_market_type, 'historical_scores')
    if hasattr(detect_market_type, 'score_dates'):
        delattr(detect_market_type, 'score_dates')
    if hasattr(detect_market_type, 'market_type_counts'):
        delattr(detect_market_type, 'market_type_counts')

    logger.info("HMM detector state has been reset")

def calculate_regime_score(adx, ma_slope, market_params):
    """
    Calculate market regime score based on HMM confidence.
    
    Args:
        adx: ADX value (not used but kept for API compatibility)
        ma_slope: MA slope value (not used but kept for API compatibility)
        market_params: Parameters based on market type
        
    Returns:
        Tuple of (score, details_dict)
    """
    # If HMM confidence is available, use it directly
    if 'confidence' in market_params:
        confidence = market_params['confidence']
        # Scale confidence to 0-100 range to match existing score scale
        regime_score = confidence * 100
        
        # Create regime details with HMM information
        regime_details = {
            'adx': adx,  # Keep for compatibility
            'ma_slope': ma_slope,  # Keep for compatibility
            'total_score': regime_score,
            'favorable': regime_score >= market_params.get('min_regime_score', 40),
            'market_type': market_params.get('market_type', 'neutral'),
            'hmm_confidence': confidence
        }
        
        return regime_score, regime_details
    
    # If no HMM confidence (fallback during warmup), use neutral values
    default_score = 50  # Neutral score
    regime_details = {
        'adx': adx,
        'ma_slope': ma_slope,
        'total_score': default_score,
        'favorable': default_score >= market_params.get('min_regime_score', 40),
        'market_type': market_params.get('market_type', 'neutral')
    }
    
    return default_score, regime_details

def calculate_position_size_adjustment(regime_score, market_params):
    """Calculate position size adjustment based on regime score or HMM confidence"""
    # If using HMM, adjust position size based on confidence
    if 'confidence' in market_params:
        confidence = market_params['confidence']
        # Use confidence to determine position sizing
        if confidence > 0.85:  # Very high confidence
            return config['position_sizing']['max_size_adjustment']
        elif confidence > 0.70:  # Good confidence
            return 1.2  # Moderate increase
        elif confidence > 0.50:  # Reasonable confidence
            return 1.0  # Normal position size
        else:  # Low confidence
            return config['position_sizing']['min_size_adjustment']
    
    """
    Calculate position size adjustment based on regime score.
    
    Args:
        regime_score: The market regime score (0-100)
        market_params: Parameters for the current market type
        
    Returns:
        Adjustment factor for position sizing
    """
    if not config['position_sizing']['adjust_by_regime']:
        return 1.0  # No adjustment if feature disabled
    
    # Get position sizing parameters
    min_adjustment = config['position_sizing']['min_size_adjustment']
    max_adjustment = config['position_sizing']['max_size_adjustment']
    
    # Get market type specific parameters
    min_score = market_params.get('min_regime_score', 40)
    sweet_spot_min = market_params.get('sweet_spot_min', 45)
    sweet_spot_max = market_params.get('sweet_spot_max', 65)
    
    # Return minimum size for scores below minimum threshold
    if regime_score < min_score:
        return min_adjustment
    
    # Apply inverted bell curve approach
    if regime_score >= sweet_spot_min and regime_score <= sweet_spot_max:
        # In sweet spot range - maximum position size
        return max_adjustment
    
    elif regime_score < sweet_spot_min:
        # Ramp up from minimum score to sweet spot
        normalized_score = (regime_score - min_score) / (sweet_spot_min - min_score)
        return min_adjustment + normalized_score * (max_adjustment - min_adjustment)
    
    else:  # regime_score > sweet_spot_max
        # Ramp down from sweet spot to maximum score
        high_score_adjustment = 0.9  # Still good but not maximum
        normalized_score = min(1.0, (regime_score - sweet_spot_max) / (100 - sweet_spot_max))
        return max_adjustment - normalized_score * (max_adjustment - high_score_adjustment)

def detect_market_type(df, lookback_days=20, current_date=None):
    """
    Detect market type using HMM-only approach.
    
    Args:
        df: DataFrame with price and indicator data
        lookback_days: Days to look back (not used with HMM but kept for API compatibility)
        current_date: Current date for detection window
        
    Returns:
        Tuple of (market_type, metrics, warmup_complete)
    """
    global hmm_detector
    
    # Initialize HMM detector if needed
    if hmm_detector is None:
        hmm_detector = initialize_hmm_detector()
        logger.info("HMM detector initialized")
    
    # Default to last date in DataFrame if no current_date
    if current_date is None:
        current_date = df['date'].max()
    
    # Check if we need to train or retrain
    if hmm_detector.model is None or hmm_detector.check_retrain_needed(current_date):
        logger.info("Training/retraining HMM model...")
        hmm_detector.fit(df, current_date)
    
    # Get HMM prediction
    prediction = hmm_detector.predict_regime(df, current_date)
    
    # Check if we have a valid prediction
    if prediction.get('needs_training', False):
        # Still in warmup phase
        logger.info("HMM model in warmup phase, defaulting to neutral market type")
        return 'neutral', {'classification_rationale': 'HMM in warmup phase'}, False
    
    # Format return values for backtester
    market_type = prediction['regime']
    confidence = prediction['confidence']
    
    metrics = {
        'confidence': confidence,
        'hmm_state': prediction.get('state', 0),
        'classification_rationale': f"HMM detected {market_type} regime with {confidence:.2f} confidence"
    }
    
    # Add feature metrics if available
    if 'features' in prediction:
        for feature, value in prediction['features'].items():
            metrics[feature] = value
    
    # Get successful warmup status
    warmup_complete = hmm_detector.model is not None
    
    return market_type, metrics, warmup_complete

def get_market_type_params(market_type, confidence=None):
    """Get parameters for the detected market type with confidence adjustment"""
    params = {}
    
    if market_type == 'trend_following':
        params = config['market_type']['trend_following'].copy()
    elif market_type == 'mean_reverting':
        params = config['market_type']['mean_reverting'].copy()
    else:  # neutral or unknown
        params = config['market_type']['neutral'].copy()
    
    # Add market type to parameters for use in regime scoring
    params['market_type'] = market_type
    
    # If HMM confidence is provided, add it to parameters
    if confidence is not None:
        params['confidence'] = confidence
    
    return params

#############################################################################
#                    TRADING STRATEGY FUNCTIONS                             #
#############################################################################

def is_in_trading_window(current_time):
    """Check if current time is within trading hours"""
    trading_hours = config['trading_hours']
    morning_session = (
        current_time >= trading_hours['morning_start'] and
        current_time <= trading_hours['morning_end']
    )
    afternoon_session = (
        current_time >= trading_hours['afternoon_start'] and
        current_time <= trading_hours['afternoon_end']
    )
    return morning_session or afternoon_session


def check_entry_signal(prev_row, current_market_type):
    # Long entry conditions (mean reversion) - LOOSENED
    long_mean_reversion = (
            prev_row['low'] < prev_row['lower_band'] and
            prev_row['RSI'] < 35 and  # CHANGED: Back to 35 from 30
            prev_row['volume'] > 1.5 * prev_row['avg_volume'] and  # CHANGED: 1.5x from 1.8x
            prev_row['close'] > prev_row['low'] * 1.0005  # CHANGED: Minimal bounce requirement
    )

    # Short entry conditions (mean reversion) - LOOSENED
    short_mean_reversion = (
            prev_row['high'] > prev_row['upper_band'] and
            prev_row['RSI'] > 65 and  # CHANGED: Back to 65 from 70
            prev_row['volume'] > 1.5 * prev_row['avg_volume'] and  # CHANGED: 1.5x from 1.8x
            prev_row['close'] < prev_row['high'] * 0.9995  # CHANGED: Minimal pullback requirement
    )

    # Long entry conditions (trend following) - LOOSENED
    long_trend_following = (
            prev_row['close'] > prev_row['MA'] and
            prev_row['RSI'] > 50 and
            prev_row['RSI'] < 70 and
            prev_row['MA_slope'] > 0.1 and  # CHANGED: Back to 0.1 from 0.15
            prev_row['volume'] > prev_row['avg_volume'] * 1.2  # CHANGED: Reduced volume requirement
    )

    # Short entry conditions (trend following) - LOOSENED
    short_trend_following = (
            prev_row['close'] < prev_row['MA'] and
            prev_row['RSI'] < 50 and
            prev_row['RSI'] > 30 and
            prev_row['MA_slope'] < -0.1 and  # CHANGED: Back to -0.1 from -0.15
            prev_row['volume'] > prev_row['avg_volume'] * 1.2  # CHANGED: Reduced volume requirement
    )
    # Check conditions based on market type
    if current_market_type == 'mean_reverting':
        if long_mean_reversion:
            return 'long'
        elif short_mean_reversion:
            return 'short'

    elif current_market_type == 'trend_following':
        if long_trend_following:
            return 'long'
        elif short_trend_following:
            return 'short'

    elif current_market_type == 'neutral':
        # MODIFIED: More selective criteria for neutral markets
        # Require stronger signals with additional confirmation

        # For long entries in neutral markets
        if long_mean_reversion and prev_row['close'] > prev_row['MA']:
            # Mean reversion long with price above MA as confirmation
            return 'long'
        elif long_trend_following and prev_row['low'] < prev_row['middle_band']:
            # Trend following long with price near middle band as confirmation
            return 'long'

        # For short entries in neutral markets
        elif short_mean_reversion and prev_row['close'] < prev_row['MA']:
            # Mean reversion short with price below MA as confirmation
            return 'short'
        elif short_trend_following and prev_row['high'] > prev_row['middle_band']:
            # Trend following short with price near middle band as confirmation
            return 'short'

    return None

def check_entry_signal_ml(prev_row, ml_features, ml_predictor, current_date, current_market_type='neutral'):
    """
    Check for entry signals with ML filtering based on market type.
    
    Args:
        prev_row: Previous bar data
        ml_features: ML features for prediction
        ml_predictor: MLPredictor instance
        current_date: Current date for retraining check
        current_market_type: Current detected market type
        
    Returns:
        Tuple of (signal, probability, approved)
    """
    strategy = config['strategy']
    ml_settings = config['ml']
    
    # Long entry conditions (mean reversion)
    long_mean_reversion = (
        prev_row['low'] < prev_row['lower_band'] and 
        prev_row['RSI'] < strategy['rsi_oversold'] and 
        prev_row['volume'] > strategy['volume_multiplier'] * prev_row['avg_volume']
    )
    
    # Short entry conditions (mean reversion)
    short_mean_reversion = (
        prev_row['high'] > prev_row['upper_band'] and 
        prev_row['RSI'] > strategy['rsi_overbought'] and 
        prev_row['volume'] > strategy['volume_multiplier'] * prev_row['avg_volume']
    )
    
    # Long entry conditions (trend following)
    long_trend_following = (
        prev_row['close'] > prev_row['MA'] and
        prev_row['RSI'] > 50 and
        prev_row['RSI'] < 70 and  # Not overbought
        prev_row['MA_slope'] > 0.1 and  # Positive slope
        prev_row['volume'] > strategy['volume_multiplier'] * prev_row['avg_volume']
    )
    
    # Short entry conditions (trend following)
    short_trend_following = (
        prev_row['close'] < prev_row['MA'] and
        prev_row['RSI'] < 50 and
        prev_row['RSI'] > 30 and  # Not oversold
        prev_row['MA_slope'] < -0.1 and  # Negative slope
        prev_row['volume'] > strategy['volume_multiplier'] * prev_row['avg_volume']
    )
    
    # Determine entry signal based on market type
    signal = None
    if current_market_type == 'mean_reverting':
        if long_mean_reversion:
            signal = 'long'
        elif short_mean_reversion:
            signal = 'short'
    
    elif current_market_type == 'trend_following':
        if long_trend_following:
            signal = 'long'
        elif short_trend_following:
            signal = 'short'
    
    elif current_market_type == 'neutral':
        # MODIFIED: More selective criteria for neutral markets
        # Require stronger signals with additional confirmation
        
        # For long entries in neutral markets
        if long_mean_reversion and prev_row['close'] > prev_row['MA']:
            # Mean reversion long with price above MA as confirmation
            signal = 'long'
        elif long_trend_following and prev_row['low'] < prev_row['middle_band']:
            # Trend following long with price near middle band as confirmation
            signal = 'long'
            
        # For short entries in neutral markets
        elif short_mean_reversion and prev_row['close'] < prev_row['MA']:
            # Mean reversion short with price below MA as confirmation
            signal = 'short'
        elif short_trend_following and prev_row['high'] > prev_row['middle_band']:
            # Trend following short with price near middle band as confirmation
            signal = 'short'
    
    if signal:
        # Check for ML approval if ML is enabled and model is trained
        if ml_settings['enable'] and ml_predictor.model is not None:
            # Check if retraining is needed
            if ml_predictor.check_retrain_needed(current_date):
                logger.info(f"ML model retraining needed. Last training: {ml_predictor.last_training_date}")
                # Retraining will be handled in the main loop
            
            # Update trade type in features
            ml_features['trade_type'] = 1 if signal == 'long' else 0
            
            # Get ML prediction
            probability, approved = ml_predictor.predict_trade_success(ml_features)
            
            # Adjust approval based on HMM confidence if available
            if 'hmm_confidence' in ml_features and ml_settings.get('use_hmm_features', True):
                hmm_confidence = ml_features['hmm_confidence']
                hmm_weight = ml_settings.get('hmm_confidence_weight', 0.3)
                
                # Weighted combination of ML probability and HMM confidence
                combined_probability = (probability * (1 - hmm_weight)) + (hmm_confidence * hmm_weight)
                
                # Only approve if combined probability exceeds threshold
                approved = combined_probability >= ml_settings['prediction_threshold']
                
                logger.debug(f"ML prob: {probability:.2f}, HMM conf: {hmm_confidence:.2f}, Combined: {combined_probability:.2f}, Approved: {approved}")
                
                # Return signal with combined probability
                return signal, combined_probability, approved
            
            # Return signal with ML information (original logic)
            return signal, probability, approved
        else:
            # ML not enabled or model not trained yet - approve all trades
            return signal, 0.5, True
    
    # No signal
    return None, 0.0, False

def check_exit_conditions(row, position, entry_price, stop_loss, profit_target, bars_held,
                          highest_price=None, lowest_price=None, atr=None, market_type='neutral'):
    """Check if position should be exited based on market type-specific conditions"""
    risk_settings = config['risk']

    # Use default max bars
    max_bars = risk_settings.get('max_bars_held', 16)
    trailing_mult = risk_settings.get('trailing_stop_atr_multiplier', 1.5)

    exit_data = {'exit': False}

    # Update highest/lowest prices if provided
    if highest_price is not None and lowest_price is not None:
        # Track the highest and lowest prices since entry
        highest_price = max(highest_price, row['high'])
        lowest_price = min(lowest_price, row['low'])
        exit_data['highest_price'] = highest_price
        exit_data['lowest_price'] = lowest_price

    # Determine if we should use trailing stops - ENHANCED APPROACH
    use_trailing = False

    # Lower activation thresholds and market-specific logic
    if position > 0:  # Long position
        if market_type == 'trend_following':
            # Trend-following trades should trail earlier
            if highest_price > entry_price * 1.0075:  # Reduced from 1.01 to 0.75%
                use_trailing = True
        elif market_type == 'mean_reverting':
            # Mean-reverting trades need more profit before trailing
            if highest_price > entry_price * 1.01:  # Keep at 1%
                use_trailing = True
        else:  # neutral market
            if highest_price > entry_price * 1.009:  # Slightly lower at 0.9%
                use_trailing = True

        # Also activate after a minimum number of bars regardless of profit
        if bars_held >= 8 and highest_price > entry_price * 1.005:  # After 8 bars with 0.5% profit
            use_trailing = True
    else:  # Short position
        if market_type == 'trend_following':
            # Trend-following trades should trail earlier
            if lowest_price < entry_price * 0.9925:  # Reduced from 0.99 to 0.75%
                use_trailing = True
        elif market_type == 'mean_reverting':
            # Mean-reverting trades need more profit before trailing
            if lowest_price < entry_price * 0.99:  # Keep at 1%
                use_trailing = True
        else:  # neutral market
            if lowest_price < entry_price * 0.991:  # Slightly lower at 0.9%
                use_trailing = True

        # Also activate after a minimum number of bars regardless of profit
        if bars_held >= 8 and lowest_price < entry_price * 0.995:  # After 8 bars with 0.5% profit
            use_trailing = True

    # Apply trailing stop if enabled and conditions are met
    if (risk_settings.get('enable_trailing_stop', False) and
            use_trailing and
            atr is not None and
            highest_price is not None and
            lowest_price is not None):

        # Calculate trailing stop with minimum cushion
        if position > 0:  # Long position
            # Ensure minimum cushion of 0.5% from highest price
            min_cushion = highest_price * 0.005
            trailing_stop = highest_price - max(trailing_mult * atr, min_cushion)

            # Never move stop loss below entry price for trades in significant profit
            if highest_price > entry_price * 1.015:  # Reduced from 2% to 1.5% profit
                trailing_stop = max(trailing_stop, entry_price)  # Ensure we don't go below entry

            # Use the higher of initial stop and trailing stop
            stop_loss = max(stop_loss, trailing_stop)

        else:  # Short position
            # Ensure minimum cushion of 0.5% from lowest price
            min_cushion = lowest_price * 0.005
            trailing_stop = lowest_price + max(trailing_mult * atr, min_cushion)

            # Never move stop loss above entry price for trades in significant profit
            if lowest_price < entry_price * 0.985:  # Reduced from 2% to 1.5% profit
                trailing_stop = min(trailing_stop, entry_price)  # Ensure we don't go above entry

            # Use the lower of initial stop and trailing stop
            stop_loss = min(stop_loss, trailing_stop)

        exit_data['updated_stop_loss'] = stop_loss

    # Check exit conditions - UNCHANGED
    if position > 0:  # Long position
        if row['low'] <= stop_loss:
            exit_data['exit'] = True
            exit_data['exit_price'] = stop_loss
            exit_data['exit_reason'] = 'trailing_stop' if (use_trailing and
                                                           stop_loss > entry_price - risk_settings[
                                                               'atr_stop_multiplier'] * atr) else 'stop_loss'
        elif row['high'] >= profit_target:
            exit_data['exit'] = True
            exit_data['exit_price'] = profit_target
            exit_data['exit_reason'] = 'profit_target'
        elif bars_held >= max_bars:
            exit_data['exit'] = True
            exit_data['exit_price'] = row['close']
            exit_data['exit_reason'] = 'time_exit'
    else:  # Short position
        if row['high'] >= stop_loss:
            exit_data['exit'] = True
            exit_data['exit_price'] = stop_loss
            exit_data['exit_reason'] = 'trailing_stop' if (use_trailing and
                                                           stop_loss < entry_price + risk_settings[
                                                               'atr_stop_multiplier'] * atr) else 'stop_loss'
        elif row['low'] <= profit_target:
            exit_data['exit'] = True
            exit_data['exit_price'] = profit_target
            exit_data['exit_reason'] = 'profit_target'
        elif bars_held >= max_bars:
            exit_data['exit'] = True
            exit_data['exit_price'] = row['close']
            exit_data['exit_reason'] = 'time_exit'

    return exit_data

def calculate_position_size(account_value, atr, position_size_adj):
    """Calculate appropriate position size based on risk and margin"""
    risk = config['risk']
    account = config['account']
    
    # Risk-based sizing with adjustment
    risk_per_contract = atr * risk['contract_multiplier']
    risk_based_contracts = max(1, int((account_value * risk['risk_per_trade'] * position_size_adj) / risk_per_contract))
    
    # Margin-based sizing
    margin_based_contracts = int(account_value / account['initial_margin'])
    
    # Apply transaction costs
    temp_num_contracts = min(risk_based_contracts, margin_based_contracts)
    adjusted_account = account_value - account['transaction_cost'] * temp_num_contracts
    final_margin_contracts = int(adjusted_account / account['initial_margin'])
    
    # Use most conservative value with a cap at 100 contracts
    return min(risk_based_contracts, final_margin_contracts, 100)
