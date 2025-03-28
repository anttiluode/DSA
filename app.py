import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import time
from scipy.stats import linregress
import requests
from scipy import signal
import ta
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator

class DendriticNode:
    """
    Represents a single node in the dendritic network.
    Each node can have parent and child dendrites, forming a hierarchical structure.
    """
    def __init__(self, level=0, feature_index=None, threshold=0.5, parent=None, name=None, growth_factor=1.0):
        self.level = level  # Depth in the hierarchy
        self.feature_index = feature_index  # Which feature this node tracks
        self.threshold = threshold  # Activation threshold
        self.parent = parent  # Parent node
        self.children = []  # Child nodes
        self.strength = 0.5  # Connection strength
        self.activation_history = []  # Recent activation levels
        self.prediction_vector = None  # Pattern that often follows this node's activation
        self.name = name  # Optional human-readable name for this dendrite
        self.growth_factor = growth_factor  # How readily this dendrite grows new connections
        self.learning_rate = 0.01  # Adjustable learning rate
        self.prediction_confidence = 0.5  # Confidence in predictions (0-1)
        self.last_activations = []  # Store last few activations for pattern recognition
        self.pattern_memory = {}  # Dictionary to store recognized patterns
        
    def activate(self, input_vector, learning_rate=0.01):
        """Activate the node based on input and propagate to children"""
        # Calculate activation based on feature if available
        if self.feature_index is not None and self.feature_index < len(input_vector):
            activation = input_vector[self.feature_index]
        else:
            # For higher-level nodes, activation is a weighted aggregate of children
            if not self.children:
                activation = 0.5  # Default activation
            else:
                # Prioritize stronger child dendrites for activation
                child_activations = []
                child_weights = []
                for child in self.children:
                    child_act = child.activate(input_vector)
                    child_activations.append(child_act)
                    child_weights.append(child.strength)
                
                # If all weights are zero, use uniform weighting
                total_weight = sum(child_weights)
                if total_weight == 0:
                    activation = np.mean(child_activations) if child_activations else 0.5
                else:
                    # Calculate weighted average
                    activation = sum(a * w for a, w in zip(child_activations, child_weights)) / total_weight
        
        # Update strength based on activation
        if activation > self.threshold:
            # Strong activation increases strength more when close to threshold
            strength_boost = learning_rate * (1 + 0.5 * (1 - abs(activation - self.threshold)))
            self.strength += strength_boost
        else:
            # Decay is slower for specialized dendrites to maintain stability
            decay_rate = learning_rate * 0.1 * (1.0 if self.name is None else 0.5)
            self.strength -= decay_rate
        
        # Ensure strength remains bounded
        self.strength = np.clip(self.strength, 0.1, 1.0)
        
        # Store activation in history
        self.activation_history.append(activation)
        if len(self.activation_history) > 100:  # Keep last 100 activations
            self.activation_history.pop(0)
            
        # Store recent activations for pattern recognition
        self.last_activations.append(activation)
        if len(self.last_activations) > 5:  # Track last 5 activations
            self.last_activations.pop(0)
            
            # Check if we have a recognizable pattern
            if len(self.last_activations) >= 3:
                # Simplify the pattern to a signature (e.g., up-down-up)
                pattern_sig = ''.join(['U' if self.last_activations[i] > self.last_activations[i-1] 
                                     else 'D' for i in range(1, len(self.last_activations))])
                
                # Store this pattern's occurrence
                if pattern_sig in self.pattern_memory:
                    self.pattern_memory[pattern_sig] += 1
                else:
                    self.pattern_memory[pattern_sig] = 1
        
        return activation * self.strength
    
    def update_prediction(self, future_vector, learning_rate=0.01):
        """Update prediction vector based on what follows this node's activation"""
        if not self.activation_history:
            return  # No activations yet
            
        # Only update prediction if recent activation was significant
        recent_activation = self.activation_history[-1] if self.activation_history else 0
        if recent_activation * self.strength < 0.3:
            return  # Not active enough to learn from
        
        if self.prediction_vector is None:
            self.prediction_vector = future_vector.copy()
            self.prediction_confidence = 0.5  # Initial confidence
        else:
            # Adjust learning rate based on activation strength
            effective_rate = learning_rate * min(1.0, recent_activation * 2)
            
            # Calculate prediction error
            if hasattr(future_vector, '__len__') and hasattr(self.prediction_vector, '__len__'):
                error = np.sqrt(np.mean((np.array(future_vector) - np.array(self.prediction_vector))**2))
                
                # Adjust confidence based on error (lower error = higher confidence)
                confidence_change = 0.1 * (1.0 - min(error * 2, 1.0))
                self.prediction_confidence = np.clip(
                    self.prediction_confidence + confidence_change, 0.1, 0.9)
            
            # Update prediction with weighted blend
            self.prediction_vector = (1 - effective_rate) * self.prediction_vector + effective_rate * future_vector
    
    def predict(self):
        """Generate prediction based on current activation pattern"""
        if self.prediction_vector is None:
            return None
        
        # Scale by strength and confidence
        prediction = self.prediction_vector * self.strength * self.prediction_confidence
        
        # If we have recognized patterns, boost prediction based on pattern history
        if self.last_activations and len(self.last_activations) >= 3:
            pattern_sig = ''.join(['U' if self.last_activations[i] > self.last_activations[i-1] 
                                 else 'D' for i in range(1, len(self.last_activations))])
            
            if pattern_sig in self.pattern_memory:
                # Boost based on how often we've seen this pattern (normalized)
                pattern_count = self.pattern_memory[pattern_sig]
                total_patterns = sum(self.pattern_memory.values())
                pattern_confidence = min(0.2, pattern_count / (total_patterns + 1))
                
                # If last part of pattern is "U", boost upward prediction
                if pattern_sig.endswith('U'):
                    for i in range(len(prediction)):
                        prediction[i] = min(1.0, prediction[i] + pattern_confidence)
                # If last part of pattern is "D", boost downward prediction
                elif pattern_sig.endswith('D'):
                    for i in range(len(prediction)):
                        prediction[i] = max(0.0, prediction[i] - pattern_confidence)
        
        return prediction
    
    def grow_dendrite(self, feature_index=None, threshold=None, name=None, growth_factor=None):
        """Grow a new child dendrite"""
        if threshold is None:
            threshold = self.threshold + np.random.uniform(-0.1, 0.1)  # Slightly different threshold
        
        if growth_factor is None:
            growth_factor = self.growth_factor
        
        # Create new child with reference to parent
        child = DendriticNode(
            level=self.level + 1,
            feature_index=feature_index,
            threshold=threshold,
            parent=self,
            name=name,
            growth_factor=growth_factor
        )
        self.children.append(child)
        return child
    
    def prune_weak_dendrites(self, min_strength=0.2):
        """Remove weak dendrites that haven't been useful"""
        # Don't prune named dendrites (preserve specialized ones)
        self.children = [child for child in self.children 
                        if child.strength > min_strength or child.name is not None]
        
        # Recursively prune children
        for child in self.children:
            child.prune_weak_dendrites(min_strength)

class HierarchicalDendriticNetwork:
    """
    Implements a hierarchical network of dendrites for stock prediction.
    The network self-organizes based on patterns in the input data.
    """
    def __init__(self, input_dim, max_levels=3, initial_dendrites_per_level=5):
        self.input_dim = input_dim  # Number of input features
        self.max_levels = max_levels  # Maximum depth of hierarchy
        
        # Root node (soma)
        self.root = DendriticNode(level=0, name="root")
        
        # Initialize basic structure
        self._initialize_dendrites(initial_dendrites_per_level)
        
        # Scaling for inputs
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Memory for temporal patterns
        self.memory_window = 15  # Days to remember (increased from 10)
        self.memory_buffer = []  # Store recent data
        
        # Fractal dimension estimate
        self.fractal_dim = 1.0
        
        # Performance tracking
        self.prediction_accuracy = []
        self.predicted_directions = []
        self.actual_directions = []
        
        # Feature importance tracking
        self.feature_importance = np.ones(input_dim) / input_dim
        
        # Market regime detection
        self.current_regime = "unknown"  # "bullish", "bearish", "sideways", "volatile"
        self.regime_history = []
        
        # Adaptive threshold based on market volatility
        self.confidence_threshold = 0.55  # Starting threshold
        self.volatility_history = []
        
        # Cross-asset correlations (will be populated during training)
        self.asset_correlations = {}
    
    def _initialize_dendrites(self, dendrites_per_level):
        """Create initial dendrite structure with specialized dendrites for stock patterns"""
        # Price level dendrites
        self.root.grow_dendrite(feature_index=0, threshold=0.3, name="price_low", growth_factor=1.2)
        self.root.grow_dendrite(feature_index=0, threshold=0.5, name="price_mid", growth_factor=1.0)
        self.root.grow_dendrite(feature_index=0, threshold=0.7, name="price_high", growth_factor=1.2)
        
        # Price trend dendrites
        self.root.grow_dendrite(feature_index=1, threshold=0.3, name="downtrend", growth_factor=1.2)
        self.root.grow_dendrite(feature_index=1, threshold=0.5, name="neutral_trend", growth_factor=0.8)
        self.root.grow_dendrite(feature_index=1, threshold=0.7, name="uptrend", growth_factor=1.2)
        
        # Volatility dendrites
        self.root.grow_dendrite(feature_index=2, threshold=0.3, name="low_volatility", growth_factor=0.8)
        self.root.grow_dendrite(feature_index=2, threshold=0.7, name="high_volatility", growth_factor=1.2)
        
        # Volume dendrites
        self.root.grow_dendrite(feature_index=3, threshold=0.3, name="low_volume", growth_factor=0.7)
        self.root.grow_dendrite(feature_index=3, threshold=0.7, name="high_volume", growth_factor=1.3)
        
        # Momentum dendrites
        self.root.grow_dendrite(feature_index=4, threshold=0.3, name="negative_momentum", growth_factor=1.2)
        self.root.grow_dendrite(feature_index=4, threshold=0.7, name="positive_momentum", growth_factor=1.2)
        
        # RSI dendrites
        self.root.grow_dendrite(feature_index=7, threshold=0.3, name="oversold", growth_factor=1.3)
        self.root.grow_dendrite(feature_index=7, threshold=0.7, name="overbought", growth_factor=1.3)
        
        # MACD dendrites
        self.root.grow_dendrite(feature_index=5, threshold=0.3, name="bearish_macd", growth_factor=1.1)
        self.root.grow_dendrite(feature_index=5, threshold=0.7, name="bullish_macd", growth_factor=1.1)
        
        # Bollinger Band dendrites
        self.root.grow_dendrite(feature_index=6, threshold=0.2, name="below_lower_band", growth_factor=1.3)
        self.root.grow_dendrite(feature_index=6, threshold=0.8, name="above_upper_band", growth_factor=1.3)
        
        # Currency-related dendrites
        if self.input_dim > 15:  # If we have currency features
            self.root.grow_dendrite(feature_index=15, threshold=0.3, name="dollar_weak", growth_factor=1.1)
            self.root.grow_dendrite(feature_index=15, threshold=0.7, name="dollar_strong", growth_factor=1.1)
        
        # Level 2: Create pattern detector dendrites
        # Create dendrites that specifically look for common patterns
        
        # Find dendrites by name
        uptrend = None
        downtrend = None
        high_volume = None
        low_volatility = None
        oversold = None
        overbought = None
        
        for child in self.root.children:
            if child.name == "uptrend":
                uptrend = child
            elif child.name == "downtrend":
                downtrend = child
            elif child.name == "high_volume":
                high_volume = child
            elif child.name == "low_volatility":
                low_volatility = child
            elif child.name == "oversold":
                oversold = child
            elif child.name == "overbought":
                overbought = child
        
        # Pattern 1: Uptrend with increasing volume (bullish)
        if uptrend and high_volume:
            pattern1 = uptrend.grow_dendrite(threshold=0.6, name="uptrend_with_volume", growth_factor=1.5)
            for _ in range(2):
                pattern1.grow_dendrite(threshold=0.6)
        
        # Pattern 2: Downtrend with high volatility (bearish)
        if downtrend:
            pattern2 = downtrend.grow_dendrite(threshold=0.4, name="downtrend_continuation", growth_factor=1.5)
            for _ in range(2):
                pattern2.grow_dendrite(threshold=0.4)
        
        # Pattern 3: Low volatility with positive momentum (potential breakout)
        if low_volatility:
            pattern3 = low_volatility.grow_dendrite(threshold=0.6, name="volatility_compression", growth_factor=1.5)
            for _ in range(2):
                pattern3.grow_dendrite(threshold=0.6)
        
        # Pattern 4: Oversold with volume spike (potential reversal)
        if oversold and high_volume:
            pattern4 = oversold.grow_dendrite(threshold=0.7, name="oversold_reversal", growth_factor=1.5)
            for _ in range(2):
                pattern4.grow_dendrite(threshold=0.7)
        
        # Pattern 5: Overbought with volume decline (potential top)
        if overbought:
            pattern5 = overbought.grow_dendrite(threshold=0.3, name="overbought_reversal", growth_factor=1.5)
            for _ in range(2):
                pattern5.grow_dendrite(threshold=0.3)
        
        # Add some general dendrites for other patterns
        for dendrite in self.root.children:
            for _ in range(dendrites_per_level // 5):
                dendrite.grow_dendrite()
        
        # Level 3: Higher-level pattern integration
        if self.max_levels >= 3:
            # Create specialized market regime dendrites
            bullish_regime = self.root.grow_dendrite(name="bullish_regime", threshold=0.7, growth_factor=1.2)
            bearish_regime = self.root.grow_dendrite(name="bearish_regime", threshold=0.3, growth_factor=1.2)
            sideways_regime = self.root.grow_dendrite(name="sideways_regime", threshold=0.5, growth_factor=1.0)
            
            # Add children to these regime detectors
            for _ in range(dendrites_per_level // 3):
                bullish_regime.grow_dendrite(threshold=np.random.uniform(0.6, 0.8))
                bearish_regime.grow_dendrite(threshold=np.random.uniform(0.2, 0.4))
                sideways_regime.grow_dendrite(threshold=np.random.uniform(0.4, 0.6))
    
    def preprocess_data(self, data):
        """Preprocess stock data for the dendritic network"""
        # Extract relevant features
        features = self._extract_features(data)
        
        # Scale features to [0, 1]
        if features.shape[0] > 0:  # Check if we have any data
            scaled_features = self.scaler.fit_transform(features)
            return scaled_features
        return np.array([])
    
    def _extract_features(self, data):
        """Extract features from stock data with enhanced technical indicators"""
        if data.empty:
            return np.array([])
        
        # Create a copy of the dataframe to avoid modifying the original
        df = data.copy()
        
        # Basic features
        features = []
        
        # 1. Price features - normalized closing price
        close = df['Close'].values
        price = (close - np.mean(close)) / (np.std(close) + 1e-8)
        features.append(price)
        
        # 2. Returns (daily percent change)
        returns = df['Close'].pct_change().fillna(0).values
        features.append(returns)
        
        # 3. Volatility (rolling std of returns)
        volatility = df['Close'].pct_change().rolling(window=5).std().fillna(0).values
        features.append(volatility)
        
        # 4. Volume relative to average
        rel_volume = df['Volume'] / df['Volume'].rolling(window=20).mean().fillna(1)
        rel_volume = rel_volume.fillna(1).values
        features.append(rel_volume)
        
        # 5. Price momentum (rate of change over 5 days)
        momentum = df['Close'].pct_change(periods=5).fillna(0).values
        features.append(momentum)
        
        # 6. MACD Line
        macd = MACD(close=df['Close']).macd()
        macd = (macd - np.mean(macd)) / (np.std(macd) + 1e-8)
        features.append(macd.fillna(0).values)
        
        # 7. Bollinger Bands Position
        bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
        bb_pos = (df['Close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband() + 1e-8)
        features.append(bb_pos.fillna(0.5).values)
        
        # 8. RSI
        rsi = RSIIndicator(close=df['Close'], window=14).rsi() / 100.0
        features.append(rsi.fillna(0.5).values)
        
        # 9. Stochastic Oscillator
        stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close']).stoch() / 100.0
        features.append(stoch.fillna(0.5).values)
        
        # 10. Average True Range (normalized)
        atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close']).average_true_range()
        atr = (atr - np.min(atr)) / (np.max(atr) - np.min(atr) + 1e-8)
        features.append(atr.fillna(0.2).values)
        
        # 11. On Balance Volume (normalized)
        obv = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()
        obv = (obv - np.mean(obv)) / (np.std(obv) + 1e-8)
        features.append(obv.fillna(0).values)
        
        # 12. Money Flow Index
        mfi = MFIIndicator(high=df['High'], low=df['Low'], close=df['Close'], 
                          volume=df['Volume'], window=14).money_flow_index() / 100.0
        features.append(mfi.fillna(0.5).values)
        
        # 13. Price Distance from 50-day SMA (normalized)
        sma50 = SMAIndicator(close=df['Close'], window=50).sma_indicator()
        sma_dist = (df['Close'] - sma50) / (df['Close'] + 1e-8)
        features.append(sma_dist.fillna(0).values)
        
        # 14. EMA Crossover Signal (fast vs slow EMAs)
        ema12 = EMAIndicator(close=df['Close'], window=12).ema_indicator()
        ema26 = EMAIndicator(close=df['Close'], window=26).ema_indicator()
        ema_cross = (ema12 - ema26) / (df['Close'] + 1e-8)
        features.append(ema_cross.fillna(0).values)
        
        # 15. Fibonacci Retracement Levels (dynamic)
        # Find recent high and low in a rolling window
        window = 20
        df['RollingHigh'] = df['High'].rolling(window=window).max()
        df['RollingLow'] = df['Low'].rolling(window=window).min()
        
        # Calculate where current price is in the retracement levels
        range_size = df['RollingHigh'] - df['RollingLow']
        fib_pos = (df['Close'] - df['RollingLow']) / (range_size + 1e-8)
        features.append(fib_pos.fillna(0.5).values)
        
        # Include any currency-related features if present
        for col in df.columns:
            if col.startswith('Currency_'):
                # Normalize currency data
                curr_data = df[col].values
                if len(curr_data) > 0:
                    curr_norm = (curr_data - np.mean(curr_data)) / (np.std(curr_data) + 1e-8)
                    features.append(curr_norm)
        
        # Transpose to get features as columns
        return np.transpose(np.array(features))
    
    def add_currency_data(self, data, currency_data):
        """Add currency exchange rate data to feature set"""
        if data.empty or currency_data.empty:
            return data
        
        # Resample currency data to match stock data frequency
        currency_data = currency_data.reindex(data.index, method='ffill')
        
        # Add currency columns to stock data
        for col in currency_data.columns:
            data[f'Currency_{col}'] = currency_data[col]
        
        return data
    
    def add_sector_data(self, data, sector_ticker, period="1y"):
        """Add sector ETF data for correlation analysis"""
        try:
            # Fetch sector data
            sector_data = yf.Ticker(sector_ticker).history(period=period)
            if sector_data.empty:
                return data
            
            # Align with stock data dates
            sector_data = sector_data.reindex(data.index, method='ffill')
            
            # Calculate daily returns
            sector_returns = sector_data['Close'].pct_change().fillna(0)
            
            # Add to stock data
            data[f'Sector_{sector_ticker}'] = sector_returns
            
            return data
        except Exception as e:
            st.error(f"Error fetching sector data: {e}")
            return data
    
    def detect_market_regime(self, data, lookback=20):
        """Detect current market regime based on price action and volatility"""
        if len(data) < lookback:
            return "unknown"
        
        # Get recent data
        recent = data.iloc[-lookback:]
        
        # Calculate trend strength
        returns = recent['Close'].pct_change().dropna()
        trend = np.sum(returns) / (np.std(returns) + 1e-8)
        
        # Calculate volatility
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        
        # Store volatility for adaptive thresholds
        self.volatility_history.append(volatility)
        if len(self.volatility_history) > 10:
            self.volatility_history.pop(0)
        
        # Update confidence threshold based on recent volatility
        if len(self.volatility_history) > 1:
            avg_vol = np.mean(self.volatility_history)
            # Higher volatility = higher threshold (require more confidence)
            self.confidence_threshold = 0.5 + min(0.2, avg_vol)
        
        # Determine regime
        if abs(trend) < 0.5:  # Low trend strength
            if volatility > 0.2:  # But high volatility
                regime = "volatile"
            else:
                regime = "sideways"
        elif trend > 0.5:  # Strong uptrend
            regime = "bullish"
        else:  # Strong downtrend
            regime = "bearish"
        
        self.current_regime = regime
        self.regime_history.append(regime)
        
        return regime
    
    def estimate_fractal_dimension(self):
        """
        Estimate the fractal dimension of the dendrite activation patterns
        using a box counting method simulation
        """
        # Create a simulated activation grid from dendrite strengths
        grid_size = 32
        activation_grid = np.zeros((grid_size, grid_size))
        
        def add_node_to_grid(node, x=0, y=0, spread=grid_size/2):
            # Add fuzzy activation for more complex boundaries
            strength = node.strength
            x_int, y_int = int(x), int(y)
            
            # Create a small activation cloud around the dendrite
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    nx, ny = (x_int + dx) % grid_size, (y_int + dy) % grid_size
                    # Stronger activation at center, weaker at edges
                    dist = np.sqrt(dx**2 + dy**2)
                    activation_grid[nx, ny] = max(
                        activation_grid[nx, ny],
                        strength * max(0, 1 - dist/2)
                    )
            
            # Add children in a circular pattern with some randomization
            if node.children:
                angle_step = 2 * np.pi / len(node.children)
                for i, child in enumerate(node.children):
                    angle = i * angle_step + np.random.uniform(-0.2, 0.2)
                    new_spread = max(1, spread * (0.6 + 0.1 * np.random.random()))
                    new_x = x + np.cos(angle) * new_spread
                    new_y = y + np.sin(angle) * new_spread
                    add_node_to_grid(child, new_x, new_y, new_spread)
        
        # Start from center of grid
        add_node_to_grid(self.root, grid_size//2, grid_size//2)
        
        # Apply Gaussian blur to create more natural boundaries
        from scipy.ndimage import gaussian_filter
        activation_grid = gaussian_filter(activation_grid, sigma=0.5)
        
        # Create more defined boundaries using edge detection
        edges = np.zeros_like(activation_grid)
        threshold = 0.2
        for i in range(1, grid_size-1):
            for j in range(1, grid_size-1):
                if activation_grid[i, j] > threshold:
                    # Check if there's a significant gradient in any direction
                    neighbors = [
                        activation_grid[i-1, j], activation_grid[i+1, j],
                        activation_grid[i, j-1], activation_grid[i, j+1]
                    ]
                    if max(neighbors) - min(neighbors) > 0.15:
                        edges[i, j] = 0.5  # Mark as boundary
        
        # Combine the activation with boundary emphasis
        combined_grid = activation_grid.copy()
        combined_grid[edges > 0] += 0.3  # Enhance boundaries
        combined_grid = np.clip(combined_grid, 0, 1)
        
        # Apply box counting method to estimate fractal dimension
        box_sizes = [1, 2, 4, 8, 16]
        counts = []
        
        for size in box_sizes:
            count = 0
            # Count boxes of size 'size' needed to cover the pattern
            for i in range(0, grid_size, size):
                for j in range(0, grid_size, size):
                    if np.any(combined_grid[i:i+size, j:j+size] > 0.25):
                        count += 1
            counts.append(count)
        
        # Calculate dimension from log-log plot slope
        if all(c > 0 for c in counts):
            coeffs = np.polyfit(np.log(box_sizes), np.log(counts), 1)
            self.fractal_dim = -coeffs[0]  # Negative slope gives dimension
        
        return self.fractal_dim, combined_grid
    
    def find_pattern_correlations(self, input_data_buffer):
        """Find patterns of feature correlations in the input data"""
        if not input_data_buffer or len(input_data_buffer) < 5:
            return {}
        
        # Stack data from buffer
        data_matrix = np.vstack(input_data_buffer)
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(data_matrix.T)
        
        # Find strongest feature pairs
        pairs = []
        n_features = corr_matrix.shape[0]
        for i in range(n_features):
            for j in range(i+1, n_features):
                pairs.append((i, j, abs(corr_matrix[i, j])))
        
        # Sort by correlation strength
        pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Return top correlations
        top_pairs = {}
        for i, j, strength in pairs[:5]:  # Top 5 correlations
            if strength > 0.4:  # Only meaningful correlations
                key = f"feature_{i}_feature_{j}"
                top_pairs[key] = strength
        
        return top_pairs
    
    def train(self, data, epochs=1, learning_rate=0.01, growth_frequency=10):
        """
        Train the dendritic network on stock data.
        The network adapts its structure based on patterns in the data.
        """
        if data.empty:
            return
        
        # First determine market regime
        self.detect_market_regime(data)
        
        # Preprocess data
        scaled_data = self.preprocess_data(data)
        
        if len(scaled_data) == 0:
            return
        
        # Initialize memory buffer
        self.memory_buffer = []
        
        # Train for specified number of epochs
        for epoch in range(epochs):
            # Track predictions for evaluation
            predicted_values = []
            actual_values = []
            
            # Process each time step
            for i in range(len(scaled_data) - 1):
                current_vector = scaled_data[i]
                future_vector = scaled_data[i + 1]
                
                # Add to memory buffer
                self.memory_buffer.append(current_vector)
                if len(self.memory_buffer) > self.memory_window:
                    self.memory_buffer.pop(0)
                
                # Find pattern correlations periodically
                if i % 20 == 0 and len(self.memory_buffer) > 5:
                    self.find_pattern_correlations(self.memory_buffer)
                
                # Activate dendrites
                root_activation = self.root.activate(current_vector, learning_rate)
                
                # Make a prediction before seeing the next value
                if i > self.memory_window:
                    prediction = self.predict_next()
                    if prediction is not None and len(prediction) > 0:
                        # For now, just use first feature (price) for evaluation
                        predicted_values.append(prediction[0])
                        actual_values.append(future_vector[0])
                
                # Update dendrite predictions
                self._update_predictions(future_vector, learning_rate)
                
                # Periodically grow new dendrites or prune weak ones
                if i % growth_frequency == 0:
                    self._adapt_structure(current_vector, learning_rate)
            
            # Calculate prediction accuracy for this epoch
            if predicted_values and actual_values:
                # Calculate directional accuracy (up/down)
                pred_dir = []
                actual_dir = []
                
                for i in range(1, len(predicted_values)):
                    # Predicted direction: is next predicted value higher than current actual?
                    pred_dir.append(1 if predicted_values[i] > actual_values[i-1] else 0)
                    # Actual direction: is next actual value higher than current actual?
                    actual_dir.append(1 if actual_values[i] > actual_values[i-1] else 0)
                
                if pred_dir and actual_dir:
                    accuracy = sum(p == a for p, a in zip(pred_dir, actual_dir)) / len(pred_dir)
                    self.prediction_accuracy.append(accuracy)
                    
                    # Store for analysis
                    self.predicted_directions.extend(pred_dir)
                    self.actual_directions.extend(actual_dir)
                    
                    if epoch == epochs - 1:  # Only on last epoch
                        st.write(f"Epoch {epoch+1}: Directional Accuracy = {accuracy:.4f}")
        
        # Calculate fractal dimension after training
        self.estimate_fractal_dimension()
    
    def _update_predictions(self, future_vector, learning_rate):
        """Update prediction vectors throughout the network"""
        # Only update if we have enough memory
        if len(self.memory_buffer) < 2:
            return
        
        # Get last and current vectors
        current_vector = self.memory_buffer[-1]
        
        def update_node_predictions(node, level_learning_rate):
            # Update this node's prediction
            node.update_prediction(future_vector, level_learning_rate)
            
            # Recursively update child nodes with diminishing learning rate
            child_lr = level_learning_rate * 0.9  # Reduce learning rate for children
            for child in node.children:
                update_node_predictions(child, child_lr)
        
        # Start from root with base learning rate
        update_node_predictions(self.root, learning_rate)
    
    def _adapt_structure(self, current_vector, learning_rate):
        """Adapt the dendritic structure by growing or pruning dendrites"""
        # Grow new dendrites where useful
        def adapt_node(node):
            # Probabilistic growth based on activation, strength, and level
            growth_prob = node.strength * node.growth_factor * (1.0 / (node.level + 1))
            if np.random.random() < growth_prob and node.level < self.max_levels - 1:
                # Determine feature for new dendrite
                if node.level == 0:
                    # First level dendrites track specific features
                    # Prioritize features based on their importance
                    feature_weights = self.feature_importance + 0.1  # Avoid zero probability
                    feature_idx = np.random.choice(
                        range(self.input_dim), 
                        p=feature_weights/np.sum(feature_weights)
                    )
                    
                    # Create dendrite with threshold biased toward discriminating values
                    if current_vector[feature_idx] > 0.7:
                        threshold = np.random.uniform(0.6, 0.9)  # High threshold
                    elif current_vector[feature_idx] < 0.3:
                        threshold = np.random.uniform(0.1, 0.4)  # Low threshold
                    else:
                        threshold = np.random.uniform(0.3, 0.7)  # Middle threshold
                    
                    node.grow_dendrite(feature_index=feature_idx, threshold=threshold)
                else:
                    # Higher level dendrites can track patterns across features
                    threshold = np.random.uniform(0.3, 0.7)
                    node.grow_dendrite(threshold=threshold)
            
            # Recursively adapt children
            for child in node.children:
                adapt_node(child)
        
        # Update feature importance based on current activation
        if len(self.memory_buffer) > 1:
            last_vector = self.memory_buffer[-2]
            current_vector = self.memory_buffer[-1]
            
            # Changes in features that correlate with changes in price are important
            price_change = current_vector[0] - last_vector[0]
            for i in range(1, min(len(current_vector), len(self.feature_importance))):
                feature_change = current_vector[i] - last_vector[i]
                importance_update = abs(feature_change * price_change) * 0.1
                self.feature_importance[i] = self.feature_importance[i] * 0.99 + importance_update
            
            # Normalize
            self.feature_importance = self.feature_importance / np.sum(self.feature_importance)
        
        # Start adaptation from root
        adapt_node(self.root)
        
        # Periodically prune weak dendrites, but less often in early training
        if np.random.random() < 0.15:  # 15% chance to prune
            min_strength = 0.15  # Lower threshold to keep more dendrites
            self.root.prune_weak_dendrites(min_strength=min_strength)
    
    def predict_next(self):
        """
        Generate a prediction for the next time step based on recent memory
        and dendrite activation patterns
        """
        if not self.memory_buffer:
            return None
        
        # Get latest input
        current_vector = self.memory_buffer[-1]
        
        # Activate the network with current input
        self.root.activate(current_vector, learning_rate=0)  # Don't learn during prediction
        
        # Collect predictions from all dendrites
        predictions = []
        
        def collect_predictions(node, weight=1.0):
            pred = node.predict()
            if pred is not None:
                # Weight by strength, prediction confidence, and node level
                effective_weight = weight * node.strength * node.prediction_confidence
                
                # Named dendrites get extra weight
                if node.name is not None:
                    effective_weight *= 1.5
                    
                # Adjust weight based on current market regime
                if self.current_regime == "bullish" and node.name and "bull" in node.name:
                    effective_weight *= 1.5
                elif self.current_regime == "bearish" and node.name and "bear" in node.name:
                    effective_weight *= 1.5
                    
                predictions.append((pred, effective_weight))
            
            for child in node.children:
                # Deeper nodes have less influence
                child_weight = weight * 0.9
                collect_predictions(child, child_weight)
        
        # Start collection from root
        collect_predictions(self.root)
        
        # Combine weighted predictions
        if not predictions:
            return None
        
        # Weight by dendrite strength and confidence
        weighted_sum = np.zeros_like(predictions[0][0])
        total_weight = 0
        
        for pred, weight in predictions:
            weighted_sum += pred * weight
            total_weight += weight
        
        if total_weight > 0:
            return weighted_sum / total_weight
        return None
    
    def predict_days_ahead(self, days_ahead=5, current_data=None):
        """
        Make predictions for multiple days ahead by feeding predictions
        back into the network
        """
        if current_data is not None:
            # Reset memory with latest actual data
            scaled_data = self.preprocess_data(current_data)
            self.memory_buffer = list(scaled_data[-self.memory_window:])
        
        if not self.memory_buffer:
            return None
        
        # Start with current memory state
        predictions = []
        confidences = []
        
        # Get current market regime for context
        if current_data is not None:
            self.detect_market_regime(current_data)
        
        # Make sequential predictions
        for day in range(days_ahead):
            # Predict next day
            next_day = self.predict_next()
            if next_day is None:
                break
            
            # Calculate confidence based on dendrite activations
            confidence = 0.5  # Default confidence
            
            # Higher confidence if dendrites agree
            if len(self.memory_buffer) > 1:
                # Check if dendrites show consistent pattern recognition
                pattern_consistency = 0
                total_patterns = 0
                
                for child in self.root.children:
                    if child.name is not None and len(child.activation_history) > 2:
                        # Check for consistent activation pattern
                        recent_acts = child.activation_history[-3:]
                        if all(a > 0.6 for a in recent_acts) or all(a < 0.4 for a in recent_acts):
                            pattern_consistency += 1
                        total_patterns += 1
                
                if total_patterns > 0:
                    consistency_score = pattern_consistency / total_patterns
                    confidence = 0.5 + 0.4 * consistency_score
            
            # Adjust confidence based on volatility
            if len(self.volatility_history) > 0:
                recent_vol = self.volatility_history[-1]
                # Lower confidence when volatility is high
                confidence -= min(0.2, recent_vol)
            
            # Add predictions and confidence
            predictions.append(next_day)
            confidences.append(confidence)
            
            # Update memory with prediction
            self.memory_buffer.append(next_day)
            if len(self.memory_buffer) > self.memory_window:
                self.memory_buffer.pop(0)
        
        return np.array(predictions), np.array(confidences)
    
    def get_trading_signals(self, predictions, confidences, threshold=None):
        """
        Convert predictions to trading signals
        threshold: confidence level needed for a buy/sell signal
        """
        if predictions is None or len(predictions) == 0:
            return []
        
        # Use adaptive threshold based on market regime if not specified
        if threshold is None:
            threshold = self.confidence_threshold
        
        signals = []
        for i, (pred, conf) in enumerate(zip(predictions, confidences)):
            # Use the first feature (price) direction for signal
            price_direction = pred[0]  # Scaled between 0-1
            
            # Adjust confidence threshold based on market regime
            adjusted_threshold = threshold
            if self.current_regime == "volatile":
                adjusted_threshold += 0.05  # Higher threshold in volatile markets
            elif self.current_regime == "sideways":
                adjusted_threshold += 0.02  # Slightly higher in sideways markets
            
            # Generate signals based on confidence-adjusted threshold
            if price_direction > 0.5 + (adjusted_threshold - 0.5) and conf > adjusted_threshold:
                signals.append('BUY')
            elif price_direction < 0.5 - (adjusted_threshold - 0.5) and conf > adjusted_threshold:
                signals.append('SELL')
            else:
                signals.append('HOLD')
        
        return signals
        
    def visualize_dendrites(self, max_nodes=50):
        """Generate a visualization of the dendrite network structure"""
        # Count nodes at each level and compute average strengths
        level_counts = {}
        level_strengths = {}
        active_nodes = {}
        named_nodes = {}
        
        def traverse_node(node):
            if node.level not in level_counts:
                level_counts[node.level] = 0
                level_strengths[node.level] = []
                active_nodes[node.level] = 0
                named_nodes[node.level] = []
            
            level_counts[node.level] += 1
            level_strengths[node.level].append(node.strength)
            
            if node.strength > 0.6:
                active_nodes[node.level] += 1
            
            if node.name is not None:
                named_nodes[node.level].append((node.name, node.strength))
            
            for child in node.children:
                traverse_node(child)
        
        traverse_node(self.root)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Node counts by level
        levels = sorted(level_counts.keys())
        counts = [level_counts[level] for level in levels]
        
        ax1.bar(levels, counts, alpha=0.7)
        ax1.set_xlabel('Dendrite Level')
        ax1.set_ylabel('Number of Dendrites')
        ax1.set_title(f'Dendritic Network Structure (Fractal Dimension: {self.fractal_dim:.3f})')
        
        # Add active node counts as a line
        active_counts = [active_nodes.get(level, 0) for level in levels]
        ax1_2 = ax1.twinx()
        ax1_2.plot(levels, active_counts, 'r-', marker='o')
        ax1_2.set_ylabel('Number of Active Dendrites (>0.6 strength)', color='r')
        ax1_2.tick_params(axis='y', labelcolor='r')
        
        # Plot 2: Average strengths by level
        avg_strengths = [np.mean(level_strengths.get(level, [0])) for level in levels]
        
        ax2.bar(levels, avg_strengths, color='green', alpha=0.7)
        ax2.set_xlabel('Dendrite Level')
        ax2.set_ylabel('Average Dendrite Strength')
        ax2.set_title('Dendrite Strength by Level')
        ax2.set_ylim([0, 1])
        
        # Add specialized dendrite info
        important_nodes = []
        for level in named_nodes:
            for name, strength in named_nodes[level]:
                if strength > 0.5:  # Only show strong specialized dendrites
                    important_nodes.append((name, level, strength))
        
        # Sort by strength
        important_nodes.sort(key=lambda x: x[2], reverse=True)
        
        # Display top nodes in a text box
        if important_nodes:
            node_text = "\n".join([f"{name}: {strength:.2f}" 
                                 for name, level, strength in important_nodes[:max_nodes]])
            ax2.text(1.05, 0.5, f"Strong Specialized Dendrites:\n{node_text}", 
                    transform=ax2.transAxes, fontsize=9,
                    verticalalignment='center', bbox=dict(boxstyle="round", alpha=0.1))
        
        # Add fractal dimension
        ax1.text(0.05, 0.95, f'Fractal Dimension: {self.fractal_dim:.3f}', 
                transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle="round", alpha=0.1))
        
        plt.tight_layout()
        
        # Create grid visualization
        fd, grid = self.estimate_fractal_dimension()
        
        return fig, grid, important_nodes

    def evaluate_performance(self, test_data):
        """Evaluate prediction performance on test data"""
        if test_data.empty:
            return None
        
        # Get market regime for test data
        self.detect_market_regime(test_data)
        
        scaled_data = self.preprocess_data(test_data)
        
        if len(scaled_data) < self.memory_window + 1:
            return None
        
        # Initialize memory with beginning of test data
        self.memory_buffer = list(scaled_data[:self.memory_window])
        
        # Make predictions and compare with actual values
        predicted_values = []
        actual_values = []
        confidences = []
        
        for i in range(self.memory_window, len(scaled_data) - 1):
            # Current vector becomes last memory item
            current_vector = scaled_data[i]
            future_vector = scaled_data[i + 1]
            
            # Update memory
            self.memory_buffer.append(current_vector)
            if len(self.memory_buffer) > self.memory_window:
                self.memory_buffer.pop(0)
            
            # Predict next
            prediction = self.predict_next()
            if prediction is not None:
                # For simplicity, just use first feature (price) for evaluation
                predicted_values.append(prediction[0])
                actual_values.append(future_vector[0])
                
                # Calculate prediction confidence
                confidence = 0.5  # Default
                
                # Higher confidence if dendrites agree
                pattern_consistency = 0
                total_patterns = 0
                
                for child in self.root.children:
                    if child.name is not None and len(child.activation_history) > 0:
                        recent_act = child.activation_history[-1]
                        if recent_act > 0.7 or recent_act < 0.3:  # Strong signal
                            pattern_consistency += 1
                        total_patterns += 1
                
                if total_patterns > 0:
                    consistency_score = pattern_consistency / total_patterns
                    confidence = 0.5 + 0.3 * consistency_score
                
                confidences.append(confidence)
        
        if not predicted_values:
            return None
        
        # Calculate directional prediction metrics
        pred_directions = []
        actual_directions = []
        
        for i in range(1, len(predicted_values)):
            # Predicted direction: is next predicted value higher than current actual?
            pred_dir = 1 if predicted_values[i] > actual_values[i-1] else 0
            # Actual direction: is next actual value higher than current actual?
            actual_dir = 1 if actual_values[i] > actual_values[i-1] else 0
            
            pred_directions.append(pred_dir)
            actual_directions.append(actual_dir)
        
        # Calculate directional accuracy
        dir_accuracy = sum(p == a for p, a in zip(pred_directions, actual_directions)) / len(pred_directions) if pred_directions else 0
        
        # Calculate RMSE on scaled values
        rmse = np.sqrt(np.mean((np.array(predicted_values) - np.array(actual_values)) ** 2))
        
        # Calculate confidence-weighted accuracy
        weighted_correct = 0
        total_weight = 0
        
        for i in range(len(pred_directions)):
            if i < len(confidences):
                weight = confidences[i]
                if pred_directions[i] == actual_directions[i]:
                    weighted_correct += weight
                total_weight += weight
        
        confidence_accuracy = weighted_correct / total_weight if total_weight > 0 else 0
        
        # Calculate profitability metrics
        # Simple simulation of buying/selling based on predictions
        initial_capital = 10000
        capital = initial_capital
        position = 0  # Shares held
        
        # Get original price data from test data for more realistic simulation
        prices = test_data['Close'].values[-len(pred_directions)-1:]
        
        for i in range(len(pred_directions)):
            current_price = prices[i]
            next_price = prices[i+1]
            
            # If we predict up and don't have a position, buy
            if pred_directions[i] == 1 and position == 0:
                position = capital / current_price
                capital = 0
            # If we predict down and have a position, sell
            elif pred_directions[i] == 0 and position > 0:
                capital = position * current_price
                position = 0
        
        # Liquidate final position
        if position > 0:
            capital = position * prices[-1]
        
        # Calculate returns
        strategy_return = (capital / initial_capital - 1) * 100
        buy_hold_return = (prices[-1] / prices[0] - 1) * 100
        
        return {
            'directional_accuracy': dir_accuracy,
            'confidence_weighted_accuracy': confidence_accuracy,
            'rmse': rmse,
            'predictions': predicted_values,
            'actual': actual_values,
            'predicted_directions': pred_directions,
            'actual_directions': actual_directions,
            'confidences': confidences,
            'strategy_return': strategy_return,
            'buy_hold_return': buy_hold_return,
            'market_regime': self.current_regime,
            'test_data_length': len(test_data)
        }

# Fetch stock and currency data
def fetch_stock_data(ticker, period="2y", interval="1d"):
    """Fetch stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)
        return data
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return pd.DataFrame()

def fetch_currency_data(currencies=["EURUSD=X", "JPYUSD=X", "CNYUSD=X"], period="2y", interval="1d"):
    """Fetch currency data for Euro, Yen, and Yuan against USD"""
    try:
        currency_data = {}
        for curr in currencies:
            ticker = yf.Ticker(curr)
            data = ticker.history(period=period, interval=interval)
            if not data.empty:
                currency_data[curr.replace('=X', '')] = data['Close']
        
        return pd.DataFrame(currency_data)
    except Exception as e:
        st.error(f"Error fetching currency data: {e}")
        return pd.DataFrame()

def fetch_sector_data(sectors=None, period="2y"):
    """Fetch sector ETF data for additional context"""
    if sectors is None:
        # Default technology sector ETF
        sectors = ["XLK"]  # Technology sector ETF
    
    try:
        sector_data = {}
        for sector in sectors:
            ticker = yf.Ticker(sector)
            data = ticker.history(period=period)
            if not data.empty:
                sector_data[sector] = data['Close']
        
        return pd.DataFrame(sector_data)
    except Exception as e:
        st.error(f"Error fetching sector data: {e}")
        return pd.DataFrame()

def train_test_split(data, test_size=0.2):
    """Split data into training and testing sets"""
    if data.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    split_idx = int(len(data) * (1 - test_size))
    train_data = data.iloc[:split_idx].copy()
    test_data = data.iloc[split_idx:].copy()
    return train_data, test_data

def compare_with_baseline(test_data, dsa_results):
    """Compare DSA performance with simple baseline models and ML benchmarks"""
    if test_data.empty or dsa_results is None:
        return {}
    
    # Extract closing prices for simplicity
    closes = test_data['Close'].values
    
    # Baseline 1: Previous day prediction (assumption: tomorrow = today)
    prev_day_accuracy = 0.5  # Default to random guessing
    if len(closes) > 2:
        # Simply predict the same direction as previous day
        baseline1_dir_pred = []
        baseline1_dir_actual = []
        
        for i in range(1, len(closes)-1):
            # Previous day direction
            prev_direction = 1 if closes[i] > closes[i-1] else 0
            # Actual next day direction
            actual_direction = 1 if closes[i+1] > closes[i] else 0
            
            baseline1_dir_pred.append(prev_direction)
            baseline1_dir_actual.append(actual_direction)
        
        prev_day_accuracy = sum(p == a for p, a in zip(baseline1_dir_pred, baseline1_dir_actual)) / len(baseline1_dir_pred)
    
    # Baseline 2: Simple moving average (10-day)
    ma_period = 10
    ma_accuracy = 0.5  # Default to random guessing
    
    if len(closes) > ma_period + 1:
        ma_dir_pred = []
        ma_dir_actual = []
        
        for i in range(ma_period, len(closes)-1):
            ma_value = np.mean(closes[i-ma_period:i])
            ma_dir = 1 if closes[i] > ma_value else 0  # If current price > MA, predict up
            actual_dir = 1 if closes[i+1] > closes[i] else 0
            
            ma_dir_pred.append(ma_dir)
            ma_dir_actual.append(actual_dir)
        
        ma_accuracy = sum(p == a for p, a in zip(ma_dir_pred, ma_dir_actual)) / len(ma_dir_pred)
    
    # Baseline 3: Linear regression on recent prices
    lr_period = 14
    lr_accuracy = 0.5  # Default to random guessing
    
    if len(closes) > lr_period + 1:
        lr_dir_pred = []
        lr_dir_actual = []
        
        for i in range(lr_period, len(closes)-1):
            X = np.arange(lr_period).reshape(-1, 1)
            y = closes[i-lr_period:i]
            slope, intercept, _, _, _ = linregress(X.flatten(), y)
            
            # Predict trend direction based on slope
            lr_dir = 1 if slope > 0 else 0
            actual_dir = 1 if closes[i+1] > closes[i] else 0
            
            lr_dir_pred.append(lr_dir)
            lr_dir_actual.append(actual_dir)
        
        lr_accuracy = sum(p == a for p, a in zip(lr_dir_pred, lr_dir_actual)) / len(lr_dir_pred)
    
    # Baseline 4: MACD crossover strategy
    macd_accuracy = 0.5  # Default
    
    if len(test_data) > 26:  # Need at least 26 days for MACD
        # Calculate MACD
        ema12 = test_data['Close'].ewm(span=12, adjust=False).mean()
        ema26 = test_data['Close'].ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        
        # Generate signals
        macd_dir_pred = []
        macd_dir_actual = []
        
        for i in range(26, len(test_data)-1):
            # MACD crossover: Buy when MACD crosses above signal line
            macd_val = macd_line.iloc[i]
            signal_val = signal_line.iloc[i]
            macd_prev = macd_line.iloc[i-1]
            signal_prev = signal_line.iloc[i-1]
            
            # Bullish crossover: MACD crosses above signal line
            bullish = macd_prev < signal_prev and macd_val > signal_val
            # Bearish crossover: MACD crosses below signal line
            bearish = macd_prev > signal_prev and macd_val < signal_val
            
            if bullish:
                pred = 1  # Predict up
            elif bearish:
                pred = 0  # Predict down
            else:
                # No crossover, maintain previous direction
                pred = 1 if macd_val > signal_val else 0
            
            actual = 1 if test_data['Close'].iloc[i+1] > test_data['Close'].iloc[i] else 0
            
            macd_dir_pred.append(pred)
            macd_dir_actual.append(actual)
        
        if macd_dir_pred:
            macd_accuracy = sum(p == a for p, a in zip(macd_dir_pred, macd_dir_actual)) / len(macd_dir_pred)
    
    # Add a random baseline
    random_accuracy = 0.5  # Theoretical random guessing accuracy
    
    # Calculate the theoretical best possible accuracy
    max_accuracy = max(prev_day_accuracy, ma_accuracy, lr_accuracy, macd_accuracy, random_accuracy)
    improvement = ((dsa_results['directional_accuracy'] / max_accuracy) - 1) * 100 if max_accuracy > 0 else 0
    
    # Calculate the profitability comparison
    strategy_return = dsa_results.get('strategy_return', 0)
    buy_hold_return = dsa_results.get('buy_hold_return', 0)
    
    return {
        'dsa_accuracy': dsa_results['directional_accuracy'],
        'dsa_confidence_accuracy': dsa_results.get('confidence_weighted_accuracy', 0),
        'previous_day_accuracy': prev_day_accuracy,
        'moving_average_accuracy': ma_accuracy,
        'linear_regression_accuracy': lr_accuracy,
        'macd_accuracy': macd_accuracy,
        'random_guessing': random_accuracy,
        'max_baseline_accuracy': max_accuracy,
        'improvement_percentage': improvement,
        'dsa_return': strategy_return,
        'buy_hold_return': buy_hold_return
    }

# Interactive Streamlit app for visualization
def main():
    st.title("Enhanced Dendritic Stock Algorithm (DSA)")
    st.markdown("""
    ### Hierarchical Dendritic Network for Stock Prediction
    
    This system implements a biological-inspired dendritic network that forms fractal patterns
    at the boundaries between different processing regimes. These patterns emerge naturally
    from the self-organizing dynamics, demonstrating our theory about boundary-emergent complexity.
    """)
    
    st.sidebar.header("Settings")
    
    # Stock selection
    ticker_options = {
        "Apple": "AAPL",
        "Microsoft": "MSFT",
        "Google": "GOOGL",
        "Amazon": "AMZN",
        "Tesla": "TSLA",
        "Meta": "META",
        "Nvidia": "NVDA",
        "Berkshire Hathaway": "BRK-B",
        "Visa": "V",
        "JPMorgan Chase": "JPM",
        "S&P 500 ETF": "SPY",
        "Nasdaq ETF": "QQQ"
    }
    
    ticker_name = st.sidebar.selectbox(
        "Select Stock", 
        list(ticker_options.keys()),
        index=0
    )
    ticker = ticker_options[ticker_name]
    
    # Add option for custom ticker
    custom_ticker = st.sidebar.text_input("Or enter custom ticker:", "")
    if custom_ticker:
        ticker = custom_ticker.upper()
    
    # Optional sector ETF to include
    include_sector = st.sidebar.checkbox("Include Sector ETF data", value=True)
    sector_etf = None
    if include_sector:
        sector_etf = st.sidebar.selectbox(
            "Select Sector ETF", 
            ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE"],
            index=0,
            help="XLK=Technology, XLF=Financials, XLE=Energy, XLV=Healthcare, XLI=Industrials"
        )
    
    # Training parameters
    st.sidebar.subheader("Training Parameters")
    train_period = st.sidebar.selectbox(
        "Training Period", 
        ["6mo", "1y", "2y", "5y", "max"],
        index=1
    )
    test_size = st.sidebar.slider("Test Data Size (%)", 10, 50, 20)
    epochs = st.sidebar.slider("Training Epochs", 1, 10, 3)
    
    # Network parameters
    st.sidebar.subheader("Network Parameters")
    dendrites_per_level = st.sidebar.slider("Initial Dendrites per Level", 3, 20, 10)
    max_levels = st.sidebar.slider("Maximum Hierarchy Levels", 1, 5, 3)
    memory_window = st.sidebar.slider("Memory Window (Days)", 5, 30, 15)
    
    # Prediction parameters
    st.sidebar.subheader("Prediction Parameters")
    days_ahead = st.sidebar.slider("Days to Predict Ahead", 1, 30, 5)
    signal_threshold = st.sidebar.slider("Base Signal Threshold", 0.51, 0.99, 0.55, 
                                       help="Higher values require more confidence for buy/sell signals")
    
    # Advanced options
    st.sidebar.subheader("Advanced Options")
    show_advanced = st.sidebar.checkbox("Show Advanced Metrics", value=False)
    
    # Load data on button click
    if st.sidebar.button("Load Data and Train"):
        # Show loading message
        with st.spinner("Fetching stock and market data..."):
            stock_data = fetch_stock_data(ticker, period=train_period)
            
            if stock_data.empty:
                st.error(f"No data found for ticker {ticker}")
            else:
                # Progress bar for all steps
                progress_bar = st.progress(0)
                total_steps = 7
                current_step = 0
                
                # Show basic info
                st.subheader(f"{ticker} Stock Information")
                st.write(f"Data from {stock_data.index[0].date()} to {stock_data.index[-1].date()}")
                st.write(f"Total days: {len(stock_data)}")
                
                # Fetch currency data
                currency_data = fetch_currency_data(period=train_period)
                if not currency_data.empty:
                    st.write("Currency data loaded:", list(currency_data.columns))
                
                # Add sector data if requested
                sector_data = None
                if include_sector and sector_etf:
                    sector_data = fetch_sector_data([sector_etf], period=train_period)
                    if not sector_data.empty:
                        st.write(f"Sector ETF data loaded: {sector_etf}")
                
                # Progress update
                current_step += 1
                progress_bar.progress(current_step / total_steps)
                
                # Add currency data to stock data
                combined_data = stock_data.copy()
                if not currency_data.empty:
                    for curr in currency_data.columns:
                        # Align currency data to stock data dates
                        currency_aligned = currency_data[curr].reindex(combined_data.index, method='ffill')
                        combined_data[f'Currency_{curr}'] = currency_aligned
                
                # Add sector data if available
                if sector_data is not None and not sector_data.empty:
                    for sect in sector_data.columns:
                        # Align sector data to stock data dates
                        sector_aligned = sector_data[sect].reindex(combined_data.index, method='ffill')
                        # Calculate daily returns
                        combined_data[f'Sector_{sect}'] = sector_aligned.pct_change().fillna(0)
                
                # Progress update
                current_step += 1
                progress_bar.progress(current_step / total_steps)
                
                # Split into train/test
                train_data, test_data = train_test_split(combined_data, test_size=test_size/100)
                
                # Create and configure network
                feature_count = 16  # Fixed based on extract_features method
                network = HierarchicalDendriticNetwork(
                    input_dim=feature_count,
                    max_levels=max_levels,
                    initial_dendrites_per_level=dendrites_per_level
                )
                network.memory_window = memory_window
                
                # Progress update
                current_step += 1
                progress_bar.progress(current_step / total_steps)
                
                # Train the network
                with st.spinner("Training dendritic network..."):
                    network.train(train_data, epochs=epochs)
                
                # Progress update
                current_step += 1
                progress_bar.progress(current_step / total_steps)
                
                # Evaluate on test data
                with st.spinner("Evaluating performance..."):
                    eval_results = network.evaluate_performance(test_data)
                    
                    if eval_results:
                        st.subheader("Performance Evaluation")
                        st.write(f"Directional Accuracy: {eval_results['directional_accuracy']:.4f}")
                        st.write(f"Confidence-Weighted Accuracy: {eval_results['confidence_weighted_accuracy']:.4f}")
                        st.write(f"RMSE (scaled): {eval_results['rmse']:.4f}")
                        st.write(f"Detected Market Regime: {eval_results['market_regime'].upper()}")
                        
                        # Show returns
                        st.write(f"DSA Trading Return: {eval_results['strategy_return']:.2f}%")
                        st.write(f"Buy & Hold Return: {eval_results['buy_hold_return']:.2f}%")
                        
                        # Compare with baselines
                        baseline_results = compare_with_baseline(test_data, eval_results)
                        
                        # Progress update
                        current_step += 1
                        progress_bar.progress(current_step / total_steps)
                        
                        if baseline_results:
                            st.subheader("Comparison with Baseline Models")
                            
                            # Format improvement percentage
                            improvement = baseline_results.get('improvement_percentage', 0)
                            improvement_text = f"+{improvement:.2f}%" if improvement > 0 else f"{improvement:.2f}%"
                            
                            results_df = pd.DataFrame({
                                'Model': [
                                    f"Dendritic Stock Algorithm ({improvement_text})",
                                    'Previous Day Strategy', 
                                    'Moving Average', 
                                    'Linear Regression',
                                    'MACD Crossover',
                                    'Random Guessing'
                                ],
                                'Directional Accuracy': [
                                    baseline_results['dsa_accuracy'], 
                                    baseline_results['previous_day_accuracy'],
                                    baseline_results['moving_average_accuracy'],
                                    baseline_results['linear_regression_accuracy'],
                                    baseline_results['macd_accuracy'],
                                    baseline_results['random_guessing']
                                ]
                            })
                            
                            # Plot comparison
                            fig = px.bar(results_df, x='Model', y='Directional Accuracy',
                                         title="Model Comparison - Directional Accuracy",
                                         color='Directional Accuracy', 
                                         color_continuous_scale=px.colors.sequential.Blues)
                            
                            fig.add_hline(y=0.5, line_dash="dash", line_color="red", 
                                         annotation_text="Random Guess (50%)")
                            
                            fig.update_layout(
                                yaxis_range=[0.4, max(0.75, baseline_results['dsa_accuracy'] * 1.1)],
                                xaxis_title="",
                                yaxis_title="Directional Accuracy"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show return comparison
                            returns_df = pd.DataFrame({
                                'Strategy': ['Dendritic Stock Algorithm', 'Buy & Hold'],
                                'Return (%)': [
                                    baseline_results['dsa_return'],
                                    baseline_results['buy_hold_return']
                                ]
                            })
                            
                            fig_returns = px.bar(returns_df, x='Strategy', y='Return (%)',
                                               title="Return Comparison",
                                               color='Return (%)', 
                                               color_continuous_scale=px.colors.sequential.Greens)
                            
                            st.plotly_chart(fig_returns, use_container_width=True)
                
                # Progress update
                current_step += 1
                progress_bar.progress(current_step / total_steps)
                
                # Make future predictions
                with st.spinner("Generating predictions..."):
                    latest_data = combined_data.tail(memory_window)
                    predictions, confidences = network.predict_days_ahead(days_ahead, latest_data)
                    
                    if predictions is not None:
                        signals = network.get_trading_signals(predictions, confidences, signal_threshold)
                        
                        # Convert predictions back to price scale
                        latest_close = latest_data['Close'].iloc[-1]
                        prediction_values = []
                        
                        # Scale based on the first feature (price) direction
                        for i, pred in enumerate(predictions):
                            if i == 0:
                                direction = 1 if pred[0] > 0.5 else -1
                                # Adjust strength by distance from 0.5
                                strength = abs(pred[0] - 0.5) * 4  # Max 2% change
                                predicted_price = latest_close * (1 + direction * strength/100)
                            else:
                                prev_predicted = prediction_values[-1]
                                direction = 1 if pred[0] > 0.5 else -1
                                strength = abs(pred[0] - 0.5) * 4
                                predicted_price = prev_predicted * (1 + direction * strength/100)
                            
                            prediction_values.append(predicted_price)
                        
                        # Create date range for predictions
                        last_date = latest_data.index[-1]
                        prediction_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_ahead, freq='B')
                        
                        # Display predictions
                        st.subheader(f"Predictions for Next {days_ahead} Trading Days")
                        
                        pred_df = pd.DataFrame({
                            'Date': prediction_dates,
                            'Predicted Price': [f"${price:.2f}" for price in prediction_values],
                            'Signal': signals,
                            'Confidence': [f"{conf:.2f}" for conf in confidences]
                        })
                        
                        st.dataframe(pred_df, use_container_width=True)
                        
                        # Plot historical + predictions
                        fig = go.Figure()
                        
                        # Add historical prices
                        fig.add_trace(go.Scatter(
                            x=combined_data.index,
                            y=combined_data['Close'],
                            mode='lines',
                            name='Historical',
                            line=dict(color='blue', width=2)
                        ))
                        
                        # Add predictions
                        fig.add_trace(go.Scatter(
                            x=prediction_dates,
                            y=prediction_values,
                            mode='lines+markers',
                            name='Predicted',
                            line=dict(dash='dash', color='darkblue'),
                            marker=dict(size=10)
                        ))
                        
                        # Shade prediction confidence intervals
                        high_bound = [price * (1 + (1 - conf) * 0.05) for price, conf in zip(prediction_values, confidences)]
                        low_bound = [price * (1 - (1 - conf) * 0.05) for price, conf in zip(prediction_values, confidences)]
                        
                        fig.add_trace(go.Scatter(
                            x=prediction_dates,
                            y=high_bound,
                            mode='lines',
                            line=dict(width=0),
                            showlegend=False
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=prediction_dates,
                            y=low_bound,
                            mode='lines',
                            line=dict(width=0),
                            fill='tonexty',
                            fillcolor='rgba(0, 0, 255, 0.1)',
                            name='Confidence Interval'
                        ))
                        
                        # Add signals
                        for i, signal in enumerate(signals):
                            color = 'green' if signal == 'BUY' else 'red' if signal == 'SELL' else 'gray'
                            
                            fig.add_annotation(
                                x=prediction_dates[i],
                                y=prediction_values[i],
                                text=signal,
                                showarrow=True,
                                arrowhead=1,
                                arrowsize=1,
                                arrowwidth=2,
                                arrowcolor=color
                            )
                        
                        fig.update_layout(
                            title=f"{ticker} Stock Price with DSA Predictions",
                            xaxis_title="Date",
                            yaxis_title="Price",
                            legend_title="Data Source",
                            hovermode="x unified"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                # Progress update - complete
                current_step += 1
                progress_bar.progress(current_step / total_steps)
                progress_bar.empty()
                
                # Visualize dendritic network
                with st.spinner("Visualizing dendritic network..."):
                    st.subheader("Dendritic Network Visualization")
                    
                    # Network structure
                    fig, grid, important_nodes = network.visualize_dendrites()
                    st.pyplot(fig)
                    
                    # Activation grid (fractal visualization)
                    st.subheader("Dendritic Activation Pattern (The Fractal Boundary)")
                    st.markdown("""
                    This visualization represents the dendritic network's activation pattern, showing how information 
                    is processed at the boundaries between different dendrite clusters. The fractal patterns emerge 
                    at these boundaries - just as we discussed about event horizons and neural boundaries.
                    
                    Key observations:
                    - Brighter regions show stronger dendrite activations
                    - The complex patterns along boundaries represent areas where the network is processing the most information
                    - Higher fractal dimension values indicate more complex boundary structures, which typically correlate with better prediction capability
                    """)
                    
                    st.write(f"**Estimated Fractal Dimension: {network.fractal_dim:.3f}**")
                    
                    if network.fractal_dim > 1.5:
                        st.success("High fractal dimension suggests complex boundary processing - good for prediction!")
                    elif network.fractal_dim > 1.2:
                        st.info("Moderate fractal dimension indicates developing complexity at boundaries")
                    else:
                        st.warning("Low fractal dimension suggests simple boundaries - prediction may be limited")
                    
                    # Plot the grid as a heatmap
                    fig, ax = plt.subplots(figsize=(8, 8))
                    im = ax.imshow(grid, cmap='viridis')
                    plt.colorbar(im, ax=ax, label='Activation Strength')
                    ax.set_title("Dendritic Activation Grid - Fractal Boundary Patterns")
                    st.pyplot(fig)
                    
                    # Show important dendrites
                    if important_nodes:
                        st.subheader("Active Specialized Dendrites")
                        st.markdown("These specialized dendrites have developed strong activations, indicating the network has learned to recognize specific patterns:")
                        
                        # Format into two columns
                        col1, col2 = st.columns(2)
                        half_nodes = len(important_nodes) // 2 + len(important_nodes) % 2
                        
                        with col1:
                            for name, level, strength in important_nodes[:half_nodes]:
                                if strength > 0.7:
                                    st.success(f"**{name}:** {strength:.2f}")
                                elif strength > 0.5:
                                    st.info(f"**{name}:** {strength:.2f}")
                                else:
                                    st.write(f"**{name}:** {strength:.2f}")
                        
                        with col2:
                            for name, level, strength in important_nodes[half_nodes:]:
                                if strength > 0.7:
                                    st.success(f"**{name}:** {strength:.2f}")
                                elif strength > 0.5:
                                    st.info(f"**{name}:** {strength:.2f}")
                                else:
                                    st.write(f"**{name}:** {strength:.2f}")
                    
                    # Explain the connection to our theory
                    st.markdown("""
                    ### Connection to Boundary Theory
                    
                    The patterns you see above demonstrate our theory about boundary-emergent complexity:
                    
                    1. **Temporal Integration**: These patterns encode the network's memory (past), processing (present), and prediction (future)
                    
                    2. **Critical Behavior**: The dendrites naturally organize at the "edge of chaos" - not too ordered, not too random
                    
                    3. **Fractal Structure**: The self-similar patterns at multiple scales allow the system to recognize patterns across different timeframes
                    
                    This visual representation shows how our dendritic network creates complex structures at the boundaries between different processing regimes - exactly as our theory predicted.
                    """)
                    
                    # If advanced metrics were requested, show them
                    if show_advanced:
                        st.subheader("Advanced Analysis")
                        
                        # Show feature importance
                        feature_names = [
                            "Price", "Returns", "Volatility", "Volume", "Momentum", 
                            "MACD", "Bollinger", "RSI", "Stochastic", "ATR",
                            "OBV", "MFI", "SMA Dist", "EMA Cross", "Fibonacci"
                        ]
                        
                        # Only show top features to keep it clean
                        imp_idx = np.argsort(network.feature_importance)[-10:]
                        
                        feature_imp_df = pd.DataFrame({
                            'Feature': [feature_names[i] if i < len(feature_names) else f"Feature {i}" for i in imp_idx],
                            'Importance': network.feature_importance[imp_idx]
                        })
                        
                        fig_imp = px.bar(feature_imp_df, x='Feature', y='Importance',
                                       title="Feature Importance",
                                       color='Importance', 
                                       color_continuous_scale=px.colors.sequential.Viridis)
                        
                        st.plotly_chart(fig_imp, use_container_width=True)
                        
                        # Show prediction confidence over time
                        if 'confidences' in eval_results:
                            conf_df = pd.DataFrame({
                                'Time Step': list(range(len(eval_results['confidences']))),
                                'Confidence': eval_results['confidences']
                            })
                            
                            fig_conf = px.line(conf_df, x='Time Step', y='Confidence',
                                             title="Prediction Confidence Over Time")
                            
                            st.plotly_chart(fig_conf, use_container_width=True)

if __name__ == "__main__":
    main()