"""
config.py - Centralized configuration for trading strategy

This module provides centralized configuration settings for the trading strategy,
including backtest parameters, risk management, and market regime detection.
"""

import os
from datetime import date, time

#############################################################################
#                        CONFIGURATION PARAMETERS                           #
#############################################################################

# Create configuration dictionary to group related parameters
config = {
    'global': {
        'random_seed': 42,        # Default random seed for reproducibility
        'use_fixed_seed': True    # Whether to use fixed random seed
    },

    'data': {
        'file_path': os.path.join('..', 'Candlestick_Data', 'MES_data', 'U19_H25.csv'),
        'start_date': '2020-01-01',
        'end_date': '2020-12-31'
    },

    'account': {
        'initial_capital': 40000,    # Starting capital in dollars
        'transaction_cost': 0.37,    # Cost per contract per trade
        'initial_margin': 1522       # Initial margin requirement per contract
    },

    'risk': {
        'risk_per_trade': 0.01,      # Risk percentage per trade (1% = 0.01)
        'atr_stop_multiplier': 2.0,  # ATR multiplier for stop loss
        'contract_multiplier': 5,    # MES is $5 per point
        'max_bars_held': 16,            # Maximum number of bars to hold a trade

        # Exit parameters
        'enable_trailing_stop': True,        # Enable trailing stops
        'trailing_stop_atr_multiplier': 1.5, # Multiplier for trailing stops
        'dynamic_target_enable': True,       # Use dynamic ATR-based targets
        'dynamic_target_atr_multiplier': 1.0     # Multiplier for ATR-based profit targets
    },

    'trading_hours': {
        'morning_start': time(9, 30),
        'morning_end': time(11, 00),
        'afternoon_start': time(15, 00),
        'afternoon_end': time(16, 00)
    },

    'strategy': {
        'bb_window': 20,             # Bollinger Bands window
        'rsi_window': 14,            # RSI calculation window
        'rsi_oversold': 35,          # RSI oversold threshold for long entries
        'rsi_overbought': 65,        # RSI overbought threshold for short entries
        'volume_multiplier': 1.5,    # Volume multiplier for entry condition
    },

    'regime': {
        'adx_window': 14,                # Period for ADX calculation
        'adx_threshold': 40,             # ADX below this is considered range-bound
        'ma_window': 50,                 # Moving average window for trend direction
        'ma_slope_window': 10,           # Period for calculating MA slope
        'ma_slope_threshold': 0.25,      # Max absolute slope value for "flat" market
        'volatility_window': 20,         # Period for calculating market volatility (keeping for data generation)
        'volatility_threshold': 1.8,     # Volatility threshold multiplier (keeping for data generation)
        'use_weighted_regime': True,     # Use weighted regime scoring
        'min_regime_score': 55,          # Minimum score required to take a trade
        'adx_weight': 0.40,              # Weight of ADX in regime score (was 0.30)
        'slope_weight': 0.60,            # Weight of MA slope in regime score (was 0.45)
        # 'vol_weight' removed - redistributed to ADX and slope
    },

    'position_sizing': {
        'adjust_by_regime': True,        # Adjust position size based on regime score
        'max_size_adjustment': 1.5,      # Maximum position size multiplier
        'min_size_adjustment': 0.8       # Minimum position size multiplier
    },

    'market_type': {
        'enable': True,                  # Enable market type classification
        'window': 10,                     # Days to look back for detection
        'update_frequency': 5,           # How often to update market type
        'trend_strength_threshold': 22,  # ADX threshold for trend detection
        'volatility_ratio_threshold': 1.0, # Volatility ratio threshold
        'momentum_threshold': 60,        # RSI threshold for momentum

        # Parameters for different market types
        'trend_following': {
            'adx_weight': 0.40,          # Updated from original value
            'slope_weight': 0.60,        # Updated from original value
            'min_regime_score': 70,
            'sweet_spot_min': 70,
            'sweet_spot_max': 90,
            'max_bars_held': 24,  # Allow longer hold time for trends
            'trailing_stop_multiplier': 1.0  # Tighter trailing for trends
        },

        'mean_reverting': {
            'adx_weight': 0.40,          # Updated from original value
            'slope_weight': 0.60,        # Updated from original value
            'min_regime_score': 40,
            'sweet_spot_min': 45,
            'sweet_spot_max': 65,
            'max_bars_held': 6,  # Shorter hold time for mean reversion
            'trailing_stop_multiplier': 2.0  # Wider trailing for mean reversion
        },

        # Neutral can stay as 50/50 split
        'neutral': {
            'adx_weight': 0.5,
            'slope_weight': 0.5,
            'min_regime_score': 50,
            'sweet_spot_min': 55,
            'sweet_spot_max': 75,
            'max_bars_held': 12,  # Medium hold time
            'trailing_stop_multiplier': 1.5
        }
    },

    'visualization': {
        'generate_png_charts': False      # Generate trade visualizations
    },

    'seasons': {
        'enable': True,              # Toggle season filter on/off
        'definitions': {
            'Q1': {
                'start_reference': (1, 15),    # January 15
                'end_reference': (3, 15),      # March 15
                'start_day': 0,                # Monday (0 = Monday, 6 = Sunday)
                'end_day': 4                   # Friday
            },
            'Q2': {
                'start_reference': (4, 15),    # April 15
                'end_reference': (6, 15),      # June 15
                'start_day': 0,                # Monday
                'end_day': 4                   # Friday
            },
            'Q3': {
                'start_reference': (7, 15),    # July 15
                'end_reference': (9, 15),      # September 15
                'start_day': 0,                # Monday
                'end_day': 4                   # Friday
            },
            'Q4': {
                'start_reference': (10, 15),   # October 15
                'end_reference': (12, 15),     # December 15
                'start_day': 0,                # Monday
                'end_day': 4                   # Friday
            }
        },
        'track_out_of_season': False   # Whether to track trades outside of seasons separately
    },

    'avoid_dates': [
        date(2024, 3, 29),    # Good Friday
        date(2024, 5, 27),    # Memorial Day
        date(2024, 12, 25),   # Christmas Day
        date(2025, 1, 1)      # New Year's Day
    ]
}

#############################################################################
#                  HIDDEN MARKOV MODEL CONFIGURATION                        #
#############################################################################

# Centralized HMM configuration - all HMM parameters can be adjusted here
hmm_config = {
    'enable': True,                  # Enable HMM-based regime detection
    'mode': 'hmm',                   # 'hmm', 'hybrid', or 'original'
    'detection_mode': 'hmm_only',    # 'hmm_only', 'rule_only', 'consensus', 'confidence_weighted'

    # Model parameters
    'n_states': 3,                   # Number of hidden states in the model
    'lookback_days': 30,             # 30Number of days of data to use for training
    'retrain_frequency': 7,          # 7Days between model retraining
    'min_samples': 200,              # 200Minimum samples required for training

    # Confidence settings
    'confidence_threshold': 0.7,     # Minimum confidence to trust HMM prediction

    # Output settings
    'visualize': True,               # Generate visualizations of HMM results
    'save_model': True,              # Save trained models for later use

    # State mapping (auto-detected during training, but can be overridden)
    'state_mapping': {
        0: 'trend_following',
        1: 'mean_reverting',
        2: 'neutral'
    }
}

# Add HMM configuration to main config
config['hmm_detector'] = hmm_config

#############################################################################
#                   MACHINE LEARNING CONFIGURATION                          #
#############################################################################

# ML configuration for enhanced backtesting
ml_config = {
    'enable': False,              # Enable ML filtering
    'model_type': 'xgboost',      # Model type: 'xgboost' or 'random_forest'
    'prediction_threshold': 0.6,  #0.6 Probability threshold for taking trades
    'retrain_frequency': 30,      #30 Days between model retraining
    'min_training_samples': 30,  #200 Minimum samples needed for training
    'warmup_trades': 30,         #100 Number of initial trades to collect before first training

    # HMM integration settings
    'use_hmm_features': True,     # Use HMM-based features in ML model
    'hmm_confidence_weight': 0.3  #0.3 Weight for HMM confidence in decision making
}

# Add ML configuration to main config
config['ml'] = ml_config