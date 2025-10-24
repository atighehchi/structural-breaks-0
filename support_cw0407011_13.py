import pandas as pd
import numpy as np


import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from scipy.signal import savgol_filter

from statsmodels.tsa.seasonal import STL

import numpy as np
from typing import List, Dict, Optional, Any, Tuple

from collections import deque


class OldPreprocessor:
    """
    Step 0: Classical preprocessing (fully self-explanatory commentary)

    Responsibilities:
      - Prepare raw input series for downstream perspectives by removing deterministic artifacts.
      - Handle universal issues that affect all perspectives, except those delegated to perspective-specific internal logic:
        1. Missing Data & Irregular Sampling: ensure continuity without bias; short gaps ffilled, long gaps resampled.
        2. Detrending: remove slow-moving trends to prevent bias in residual and threshold perspectives.
        3. Deseasonalization: remove repeating periodic patterns to avoid spurious break detection.
        4. Noise reduction / smoothing: reduce high-frequency fluctuations that can trigger false positives.
        5. Heteroskedasticity tracking (rolling std) is computed internally for perspectives if needed.
        6. Scale drift / normalization (global z-score) applied for all perspectives except fragility to preserve raw variance.

    Notes / Considerations:
      - Fragility perspective requires raw post-classical-preprocessed data without normalization.
      - Surprise perspective is not handled here; anomalies/outliers are managed separately in aggregator as modulator.
      - Missing values / irregular timestamps: only handled if gaps are meaningful (not single points).
      - Output series maintains the same length as input.
      - Designed for online or batch processing.

    Inputs:
      - series: pd.Series of floats, can be online (point-by-point) or batch.

    Outputs:
      - cleaned pd.Series, suitable for perspective scoring.
      - Optionally, can return additional metadata (rolling std, etc.) for perspective use.

    DL Integration:
      - None at this stage; outputs feed into DLPreprocessor for embeddings generation.

    Dataflow / Sequencing Logic:
      1. Handle missing data & irregular sampling (ffill short gaps, resample long gaps if needed).
      2. Denoise series using Savitzky-Golay / LOESS smoothing.
      3. Detrend series using STL decomposition or rolling mean subtraction.
      4. Remove seasonality if seasonal_period is set.
      5. Apply global z-score normalization for perspectives other than fragility.
      6. Compute rolling std for heteroskedasticity tracking.
      7. Output preprocessed series for downstream perspectives.

    Implementation Guidance for Independent Coder:
      - Each step preserves series length.
      - Maintain online capability: allow point-by-point updates where possible.
      - Ensure NaN-safe operations for missing data.
      - Window sizes, smoothing fraction, and seasonal period should be configurable.
      - Keep modular: each method can be reused independently.

    Example Usage:
        series = pd.Series([1,2,3,4,5])
        preprocessor = Preprocessor(loess_frac=0.05, seasonal_period=12)
        clean_series = preprocessor.preprocess(series)
    """

    def __init__(self, loess_frac: float = 0.05, seasonal_period: Optional[int] = None):
        self.loess_frac = loess_frac
        self.seasonal_period = seasonal_period

    def handle_missing_data(self, series: pd.Series, max_ffill_gap: int = 3) -> pd.Series:
        """
        Fill short gaps via forward fill, optionally resample longer gaps.
        Parameters:
          - max_ffill_gap: max consecutive NaNs to ffill; longer gaps remain NaN for possible resampling.
        """
        series_filled = series.copy()
        # Forward fill short gaps
        gap_mask = series_filled.isna()
        if gap_mask.any():
            # Count consecutive NaNs
            group = (~gap_mask).cumsum()
            counts = gap_mask.groupby(group).cumsum()
            series_filled = series_filled.ffill(limit=max_ffill_gap)
        # Optionally, could implement resampling for longer gaps if index is datetime
        return series_filled

    def denoise(self, series: pd.Series, window: int = 11, polyorder: int = 2) -> pd.Series:
        """
        Remove high-frequency noise using Savitzky-Golay filter.
        Parameters:
          - window: window length (must be odd and >= polyorder + 2)
          - polyorder: polynomial order
        """
        if len(series) < window:
            return series
        if window % 2 == 0:
            window += 1  # ensure odd
        denoised = savgol_filter(series.values, window_length=window, polyorder=polyorder)
        return pd.Series(denoised, index=series.index)

    def detrend(self, series: pd.Series, period: int = 20) -> pd.Series:
        """
        Remove slow-moving trends using STL decomposition.
        Parameters:
          - period: approximate seasonal/trend period for STL
        """
        if len(series) < period * 2:
            # Too short to detrend, fallback: subtract mean
            return series - series.mean()
        stl = STL(series, period=period, robust=True)
        result = stl.fit()
        return series - result.trend

    def remove_seasonality(self, series: pd.Series) -> pd.Series:
        """
        Remove seasonal component using STL if seasonal_period is set.
        """
        if self.seasonal_period is None or len(series) < self.seasonal_period * 2:
            return series
        stl = STL(series, period=self.seasonal_period, robust=True)
        result = stl.fit()
        return series - result.seasonal

    def normalize(self, series: pd.Series, skip_fragility: bool = False) -> pd.Series:
        """
        Apply global z-score normalization for all perspectives except fragility.
        """
        if skip_fragility:
            return series
        mean = series.mean()
        std = series.std()
        if std == 0:
            return series - mean  # avoid division by zero
        return (series - mean) / std

    def compute_rolling_std(self, series: pd.Series, window: int = 20) -> pd.Series:
        """
        Compute rolling standard deviation for heteroskedasticity tracking.
        """
        if len(series) < window:
            return pd.Series(np.zeros(len(series)), index=series.index)
        return series.rolling(window=window, min_periods=1).std()

    def preprocess(self, series: pd.Series, skip_normalization_for_fragility: bool = False) -> pd.Series:
        """
        Full preprocessing pipeline:
          1. Handle missing data
          2. Denoise
          3. Detrend
          4. Remove seasonality
          5. Normalize (conditionally)
          6. Optionally compute rolling std internally
        """
        clean = self.handle_missing_data(series)
        clean = self.denoise(clean)
        clean = self.detrend(clean)
        clean = self.remove_seasonality(clean)
        clean = self.normalize(clean, skip_fragility=skip_normalization_for_fragility)
        # Rolling std computed internally for perspectives if needed
        _ = self.compute_rolling_std(clean)
        return clean


#todo: merge and replace the previous preprocessor with this ONLINE one
from collections import deque

class OnlinePreprocessor:
    """
    Non-parametric online preprocessing for streaming data.
    
    Tasks:
    -------
    1. Detrending: estimates trend online using a Kalman filter.
    2. Deseasonalising: removes repetitive patterns non-parametrically
       using an adaptive online seasonal estimate (rolling quantile/median).
    3. Denoising (optional): removes high-frequency noise via exponential smoothing
       of residuals.
    
    Design choices:
    ---------------
    - All components are optional and can be disabled by setting the respective
      parameter to None.
    - No fixed assumptions about season length or trend shape.
    - Fully suitable for domain-agnostic online change-point detection.
    
    Usage:
    -------
    p = OnlinePreprocessor(detrend=True, denoise_alpha=0.2, season_window=50)
    for x in data_stream:
        x_tilde = p.update(x)
        cp_score = bocpd.update(x_tilde)
    """

    def __init__(self, detrend=True, denoise_alpha=None, season_window=None):
        self.detrend = detrend
        self.denoise_alpha = denoise_alpha
        self.season_window = season_window

        # Online state
        self.t = 0
        self.prev_denoised = None

        # Kalman filter initialization for trend
        if self.detrend:
            self.kf_x = None  # trend estimate
            self.kf_P = 1.0   # estimate covariance
            self.kf_Q = 1e-5  # process noise
            self.kf_R = 1e-2  # observation noise

        # Seasonal component buffer
        if self.season_window is not None:
            self.season_buffer = deque(maxlen=season_window)

    def _kalman_update(self, x):
        """
        Online Kalman filter for trend estimation.
        Returns the detrended value.
        """
        if self.kf_x is None:
            self.kf_x = x  # initialize
        # Prediction step
        P_pred = self.kf_P + self.kf_Q
        # Update step
        K = P_pred / (P_pred + self.kf_R)  # Kalman gain
        self.kf_x = self.kf_x + K * (x - self.kf_x)
        self.kf_P = (1 - K) * P_pred
        return x - self.kf_x

    def _adaptive_deseasonalise(self, residual):
        """
        Adaptive non-parametric deseasonalising using rolling median.
        """
        if self.season_window is None:
            return residual
        if len(self.season_buffer) == 0:
            self.season_buffer.append(residual)
            return residual
        median_season = np.median(self.season_buffer)
        self.season_buffer.append(residual)
        return residual - median_season

    def update(self, x):
        """
        Process a new observation and return the preprocessed residual.
        """
        self.t += 1
        residual = x

        # Step 1: Detrending via Kalman filter
        if self.detrend:
            residual = self._kalman_update(x)

        # Step 2: Deseasonalising non-parametrically
        residual = self._adaptive_deseasonalise(residual)

        # Step 3: Optional denoising (EMA on residuals)
        if self.denoise_alpha is not None:
            if self.prev_denoised is None:
                self.prev_denoised = residual
            else:
                self.prev_denoised = (
                    (1 - self.denoise_alpha) * self.prev_denoised
                    + self.denoise_alpha * residual
                )
            residual = residual - self.prev_denoised

        return residual


    

class DLPreprocessor:
    """
    Step 0: DL-based preprocessing (fully self-explanatory commentary)

    Responsibilities:
      - Provide cleaned point for perspectives, complementing classical preprocessing.
      - Generate embedding vector representing local temporal patterns via CNN.

    Inputs (per new point x):
      - x: scalar float, the latest observation.

    Outputs:
      - cleaned_point: float, locally adjusted via rolling buffer and CNN.
      - embedding: np.ndarray of emb_dim, capturing short-term temporal dynamics.

    Rationale / Notes:
      - Cleaned point can be used in all perspectives as needed.
      - Embeddings capture local temporal patterns.
      - Anomaly/outlier detection handled separately in aggregator.
      - Fully online: one point at a time.

    DL Integration:
      - 1D CNN with small receptive field, takes rolling buffer as input.
      - Produces emb_dim-dimensional embedding for perspectives.

    Dataflow / Sequencing Logic:
      1. Maintain rolling buffer of last `window` points.
      2. transform(x):
         a. Update buffer with x.
         b. Standardize buffer (demean / optional rolling std).
         c. Feed buffer into CNN to produce embedding.
         d. Cleaned point = last point in buffer minus local mean (residual).
      3. Return (cleaned_point, embedding).

    Implementation Guidance:
      - Buffer as list; convert to tensor for CNN input.
      - Numerical stability: handle small buffers gracefully.
      - Designed for online sequential updates.
      - emb_dim and window configurable.
    """

    class CNNEmbedder(nn.Module):
        """1D CNN for rolling buffer embeddings"""
        def __init__(self, input_len: int, emb_dim: int):
            super().__init__()
            # Two convolution layers + global average pooling
            self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            self.pool = nn.AdaptiveAvgPool1d(1)  # global avg pool
            self.fc = nn.Linear(64, emb_dim)

        def forward(self, x):
            # x shape: (batch=1, 1, input_len)
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.pool(x).squeeze(-1)  # shape: (1, 64)
            embedding = self.fc(x)        # shape: (1, emb_dim)
            return embedding

    def __init__(self, emb_dim: int = 16, window: int = 50, device: str = "cpu"):
        self.emb_dim = emb_dim
        self.window = window
        self._buffer: List[float] = []
        self.device = device

        # Initialize CNN embedder
        self.embedder = self.CNNEmbedder(input_len=self.window, emb_dim=self.emb_dim).to(self.device)
        self.embedder.eval()  # inference mode; no gradient updates

    def update_buffer(self, x: float):
        """Add new point x to rolling buffer, maintain window length"""
        self._buffer.append(x)
        if len(self._buffer) > self.window:
            self._buffer.pop(0)

    def transform(self, x: float) -> Tuple[float, np.ndarray]:
        """
        Update buffer, compute cleaned point, and generate CNN embedding.

        Parameters
        ----------
        x : float
            New observation

        Returns
        -------
        cleaned_point : float
            Locally adjusted scalar value.
        embedding : np.ndarray
            Shape (emb_dim,), CNN embedding vector.
        """
        # 1. Update buffer
        self.update_buffer(x)

        # 2. Convert buffer to tensor
        buffer_len = len(self._buffer)
        buffer_array = np.array(self._buffer, dtype=np.float32)

        # Demean over buffer for local adjustment
        mean_local = np.mean(buffer_array)
        std_local = np.std(buffer_array) if np.std(buffer_array) > 1e-6 else 1.0
        normalized_buffer = (buffer_array - mean_local) / std_local

        # 3. Prepare tensor for CNN: shape (batch=1, channel=1, seq_len)
        input_tensor = torch.tensor(
            normalized_buffer, dtype=torch.float32, device=self.device
        ).unsqueeze(0).unsqueeze(0)

        # Pad if buffer is smaller than window
        if buffer_len < self.window:
            pad_len = self.window - buffer_len
            input_tensor = nn.functional.pad(input_tensor, (pad_len, 0), "constant", 0.0)

        # 4. Forward pass through CNN
        with torch.no_grad():
            emb_tensor = self.embedder(input_tensor)
        embedding = emb_tensor.cpu().numpy().flatten()

        # 5. Cleaned point = last value in buffer minus local mean (residual)
        cleaned_point = x - mean_local

        return cleaned_point, embedding

    def update(self, x: float) -> Tuple[float, np.ndarray]:
        """
        Public update interface for DLPreprocessor (pipeline-compliant).

        Responsibilities
        ----------------
        - Accept a new raw input point x.
        - Internally call transform(x) for full preprocessing.
        - Ensure consistent output contract: (x_clean, emb).
        - Add safety checks (types, shape, dimensionality).
        - Keep design online and streaming.

        Parameters
        ----------
        x : float
            New raw input observation.

        Returns
        -------
        x_clean : float
            Cleaned / stabilized scalar value.
        emb : np.ndarray
            Embedding vector (1D, fixed dimension).
        """
        x_clean, emb = self.transform(x)

        # Safety checks: enforce proper types and shapes
        if not isinstance(x_clean, (int, float, np.floating)):
            raise TypeError(f"x_clean must be scalar float, got {type(x_clean)}")
        if not isinstance(emb, np.ndarray):
            emb = np.array(emb, dtype=np.float32)
        if emb.ndim != 1:
            raise ValueError(f"Embedding must be 1D array, got shape {emb.shape}")

        return x_clean, emb



class ShockDetector:
    """
    Step 1: Shock / outlier detection (fully self-explanatory commentary)

    Responsibilities:
      - Identify extreme deviations that might bias perspective scores.
      - Produce a probabilistic score in [0,1] for each new observation.
      - Optionally mask or downweight points in subsequent perspective updates.

    Inputs (per new point x):
      - x: scalar float.

    Outputs:
      - shock_score: float in [0,1], representing extremity.

    DL integration:
      - None in classical method; could incorporate embeddings for adaptive thresholds.

    Implementation Logic (textual plan, copied verbatim from design discussion):

      1. Maintain rolling buffer of last `window` points.
      2. Compute median and MAD (robust deviation metrics):
         - median = np.median(_buffer)
         - mad = np.median(abs(_buffer - median))
         - fallback if buffer too small or MAD==0
      3. Compute deviation of new point:
         - dev = abs(x - median)
      4. Map deviation to [0,1] shock score:
         - shock_score = min(dev / (threshold * effective_mad), 1.0)
         - effective_mad = max(MAD, dev / threshold)  # ensures extreme spikes flagged immediately
      5. Return shock_score online, one point at a time.

    Caveats / Reliability Notes:
      - Small buffer: need fallback logic
      - Zero MAD: avoid divide-by-zero
      - Window length: trade-off between responsiveness and robustness
      - Heavy tails: MAD is robust but extreme spikes saturate score
      - Ensure output always in [0,1]

    Integration notes:
      - Feeds perspectives to downweight their contribution
      - Not for SurprisePerspective
      - Optional DL embeddings could adapt thresholds dynamically

    Example usage:
        shock_detector = ShockDetector(method="MAD", threshold=3.0, window=10)
        score = shock_detector.update(0.5)
    """

    def __init__(self, method: str = "MAD", threshold: float = 3.0, window: int = 10):
        self.method = method        # method for deviation computation
        self.threshold = threshold  # threshold multiplier for score scaling
        self.window = window        # rolling buffer length
        self._buffer: List[float] = []  # store last `window` points
        self._ewma_mad: Optional[float] = None  # smoothed MAD for lag reduction
        self._follow_buffer = deque(maxlen=10)   # small lookahead buffer used for deciding persistent shift
        self._persistence_check_len = 5          # how many points after spike to check for sustained shift
        self._persistence_mean_threshold = 0.5   # how far mean must move to be considered persistent (in units of MAD)

    def update_buffer(self, x: float):
        """
        Maintain rolling buffer of last `window` points.
        Remove oldest if buffer exceeds window length.
        """
        self._buffer.append(x)
        if len(self._buffer) > self.window:
            self._buffer.pop(0)

    def update(self, x: float) -> float:
        """
        Computes shock score as before but also tags the shock as 'persistent' if
        the local mean over the next few points remains shifted. The 'persistent'
        tag is returned indirectly in self._last_was_persistent (bool).
        """
        # Maintain buffer for current calculation
        self.update_buffer(x)

        # Compute robust median & MAD
        buf = np.array(self._buffer, dtype=float)
        if len(buf) < 2:
            median = float(np.median(buf)) if len(buf) > 0 else 0.0
            mad = 1e-6
        else:
            median = float(np.median(buf))
            mad = float(np.median(np.abs(buf - median))) or 1e-6

        # EWMA smoothing on MAD using fixed alpha to stabilize
        alpha_fixed = 0.3
        if self._ewma_mad is None:
            self._ewma_mad = mad
        else:
            self._ewma_mad = alpha_fixed * mad + (1 - alpha_fixed) * self._ewma_mad

        effective_mad = max(self._ewma_mad, abs(x - median) / max(self.threshold, 1e-6))

        # Basic normalized dev -> [0,1]
        raw_shock = min(abs(x - median) / (self.threshold * effective_mad + 1e-12), 1.0)

        # Append to follow buffer (used to decide persistence)
        self._follow_buffer.append(x)

        # Default: not persistent (until we see next points)
        self._last_was_persistent = False

        # If this point is a candidate shock, check the following points when available
        # If enough future points are in buffer (we'll use a trailing window logic),
        # compute whether the mean in the next few points remains shifted by more than
        # persistence_mean_threshold * MAD -> mark persistent.
        if raw_shock > 0.8 and len(buf) >= self._persistence_check_len + 1:
            # recent baseline: compute mean just before the shock within buffer
            # find position of shock in buffer (we assume it's the last element)
            baseline_window = buf[:-self._persistence_check_len] if len(buf) > self._persistence_check_len else buf[:-1]
            if len(baseline_window) >= 1:
                baseline_mean = float(np.mean(baseline_window[-self._persistence_check_len:]))
                after_window = buf[-self._persistence_check_len:]
                after_mean = float(np.mean(after_window))
                # If after_mean is shifted compared to baseline by more than threshold*MAD => persistent
                if abs(after_mean - baseline_mean) > (self._persistence_mean_threshold * (mad + 1e-8)):
                    self._last_was_persistent = True

        # Map raw_shock to final shock_score (still in [0,1])
        self._last_shock_score = float(raw_shock)
        return float(raw_shock)


class Perspective:
    """
    Base class for all perspectives.

    Contract:
      - Must implement update(x: float, embedding: Optional[Array], shock_score: float) -> float
      - Returns score in [0,1] reflecting break likelihood for new point
      - Embeddings passed from DLPreprocessor to refine scoring
      - Designed for online sequential updates.
    """
    def update(self, x: float, embedding: Optional[np.array] = None, shock_score: float = 0.0) -> float:
        raise NotImplementedError


class SurprisePerspective(Perspective):
    """
    Step 2: Surprise / anomaly perspective (improved adaptive implementation)

    Responsibilities:
      - Detect unusually large deviations at a single point using adaptive thresholds.
      - Acts as a complementary signal to other perspectives.
      - Produces [0,1] score for break probability aggregation.

    Inputs:
      - x: scalar float, latest observation.
      - embedding: required np.ndarray, can refine adaptive thresholds.
      - shock_score: float in [0,1], downweights extreme points flagged by ShockDetector.

    Outputs:
      - score: float in [0,1], contribution to aggregated break probability.

    Dataflow and logic:
      1. Maintain rolling buffer of last `window` points.
      2. Compute adaptive thresholds from buffer (percentiles, embedding-driven adjustment).
      3. Map x's deviation from buffer distribution to a soft [0,1] score.
      4. Incorporate embedding distance to adjust surprise sensitivity.
      5. Apply capped shock_score downweight.
      6. Clamp to [0,1].
      7. Designed for online processing: one point at a time.

    Caveats / Issues:
      - Local buffer sensitivity: small buffer → noisy thresholds; large buffer → delayed detection.
      - Single-point deviations may be outliers, not true DGP breaks.
      - Edge effects at start of series.

    Reliability Measures / Mitigations:
      - Minimum buffer length before computing score.
      - Soft scoring avoids flatlining near zero.
      - Embedding-driven adaptive thresholds increase robustness to regime differences.
      - Cap on shock downweight prevents total suppression of surprise signals.
      - Optional diagnostic logging for buffer and score inspection.
      - Recognize complementary role: may weight lower in cross-perspective aggregation.

    Notes:
      - Embeddings from preprocessing are used to adapt thresholds and sensitivity.
      - Fully online, lightweight computation.
      - Percentile + soft scoring chosen over pure hard thresholds for robustness.

    Example usage:
        sp = SurprisePerspective(window=50)
        score = sp.update(0.5, embedding=np.zeros(16), shock_score=0.1)
    """

    def __init__(self, window: int = 50, diagnostic_logging: bool = False):
        self.window = window
        self._buffer: List[float] = []
        self._embedding_buffer: List[np.ndarray] = []
        self.diagnostic_logging = diagnostic_logging  # Off by default

    def update(self, x: float, embedding: np.ndarray, shock_score: float = 0.0) -> float:
        """
        Online update for new point x.
        """
        # 1. Update buffers
        self._buffer.append(x)
        self._embedding_buffer.append(embedding)
        if len(self._buffer) > self.window:
            self._buffer.pop(0)
            self._embedding_buffer.pop(0)

        # 2. Minimum buffer length check
        if len(self._buffer) < 5:
            return 0.0

        buf = np.array(self._buffer)

        # 3. Adaptive percentile thresholds
        lower_percentile = np.percentile(buf, 10)
        upper_percentile = np.percentile(buf, 90)

        # embedding-driven adjustment
        emb_var = np.mean(np.var(np.array(self._embedding_buffer), axis=0))
        sensitivity = 1.0 / (1.0 + emb_var)  # higher var → less sensitive
        lower_adj = np.percentile(buf, 5 + 5 * sensitivity)
        upper_adj = np.percentile(buf, 95 - 5 * sensitivity)

        # 4. Soft deviation score (Gaussian-like)
        median_val = np.median(buf)
        spread = np.std(buf) + 1e-8
        z = (x - median_val) / spread
        base_score = 1.0 - np.exp(-0.5 * (z ** 2))  # grows with |z|

        # emphasize if outside adaptive bounds
        if x < lower_adj or x > upper_adj:
            base_score = min(1.0, base_score * 1.5)

        # 5. Embedding distance adjustment
        emb_centroid = np.mean(np.array(self._embedding_buffer), axis=0)
        emb_dist = np.linalg.norm(embedding - emb_centroid)
        emb_factor = np.tanh(emb_dist)  # smoothly maps [0,∞) → [0,1)
        score = base_score * (0.8 + 0.2 * emb_factor)

        # 6. Apply capped shock downweight (soft gating, prevents double suppression)
        score *= max(0.5, 1.0 - shock_score)

        # 7. Clamp to [0,1]
        score = float(np.clip(score, 0.0, 1.0))

        # 8. Optional diagnostics
        if self.diagnostic_logging:
            print(f"[SurprisePerspective] x={x:.4f}, buf_len={len(buf)}, "
                  f"score={score:.4f}, lower_adj={lower_adj:.4f}, upper_adj={upper_adj:.4f}, "
                  f"emb_var={emb_var:.6f}, emb_dist={emb_dist:.4f}")

        return score



class PostProcessor:
    """
    Step 5: Post-processing (smoothing stage of the pipeline)

    Responsibilities:
      - Reduce jitter and short-term flip-flopping in the aggregated probability
        scores produced by the aggregator module.
      - Stabilize the time series of break probabilities so that downstream
        interpretation is less sensitive to noise, while still preserving
        responsiveness to genuine changes.
      - Maintain strictly probabilistic outputs (continuous values in [0,1]), not binary decisions.

    Inputs:
      - combined_scores: List[float]
          A sequence of raw combined break probabilities in [0,1],
          output directly from the Aggregator module.

    Outputs:
      - smoothed_scores: List[float]
          A stabilized version of combined_scores, after applying smoothing
          logic. Each element is still a probability in [0,1].

    Dataflow and logic:
      1. Initialize smoothing with the first raw score.
      2. Apply Exponentially Weighted Moving Average (EWMA) sequentially:
         s_t = α * x_t + (1 - α) * s_(t-1)
         where:
           - s_t = smoothed value at time t
           - x_t = raw combined score at time t
           - α = smoothing factor in (0,1], controls responsiveness
      3. Return the full sequence of smoothed scores.

    Implementation guidance:
      - α close to 1 → more responsive, less smoothing.
      - α close to 0 → more smoothing, less responsiveness.
      - EWMA preserves values in [0,1] if inputs are in [0,1].
      - Initialization: set first smoothed value = first raw score.

    Future upgrades:
      - Hysteresis logic could be added if binary outputs are required.
      - Adaptive α: smoothing factor dependent on score volatility.
      - Hybrid smoothing: combine short-term and long-term stabilization.
      - Diagnostic hooks: log raw and smoothed series for validation.

    Computational notes:
      - Complexity: O(n), single pass over the series.
      - Memory: O(n) to return smoothed list (can be O(1) in streaming mode).
    """

    def __init__(self, alpha: float = 0.3):
        """
        Parameters:
          - alpha (float): Smoothing factor in (0,1].
              * Controls trade-off between responsiveness and stability.
              * Default = 0.3 (moderate smoothing).
        """
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0,1].")
        self.alpha = alpha

    def smooth(self, combined_scores: List[float]) -> List[float]:
        """
        Apply EWMA smoothing to the input sequence of combined scores.

        Parameters:
          - combined_scores (List[float]): Raw probability scores from aggregator.

        Returns:
          - smoothed_scores (List[float]): Stabilized probability scores.

        Notes:
          - If combined_scores is empty, return an empty list.
          - First value is initialized directly from the first raw score.
          - EWMA preserves probabilistic [0,1] outputs.
        """
        if not combined_scores:
            return []

        smoothed_scores: List[float] = []

        # Step 1: Initialize with the first raw score
        s_prev = combined_scores[0]
        smoothed_scores.append(s_prev)

        # Step 2: Apply EWMA sequentially
        for x_t in combined_scores[1:]:
            s_t = self.alpha * x_t + (1 - self.alpha) * s_prev
            # Ensure output remains in [0,1] even if numerical drift occurs
            s_t = max(0.0, min(1.0, s_t))
            smoothed_scores.append(s_t)
            s_prev = s_t

        return smoothed_scores


class OutputHandler:
    """
    Step 4: Output final break probability (fully self-explanatory commentary)

    Responsibilities:
      - Maintain history of per-point scores (perspective-wise and combined).
      - Optionally alert if break probability exceeds threshold.
      - Ensure compatibility with postprocessing.

    Inputs:
      - scores: Dict[str, float], perspective-wise latest scores.
      - combined: float, aggregated score.

    Outputs:
      - stored internally; optional alerts printed.

    Dataflow and logic:
      1. Append latest scores and combined score to internal lists.
      2. Check combined against alert_threshold; if exceeded, optionally alert.
      3. Designed for online sequential operation.

    Implementation guidance:
      - Keep internal lists manageable for long series (truncate if needed).
      - Alerting can be customized to external systems.
      - Do not modify scores; only record and optionally alert.
    """
    def __init__(self, alert_threshold: float = 0.8):
        self.alert_threshold = alert_threshold
        self.scores: List[Dict[str, float]] = []
        self.combined_scores: List[float] = []

    def record(self, scores: Dict[str, float], combined: float):
        self.scores.append(scores)
        self.combined_scores.append(combined)
        if combined >= self.alert_threshold:
            # placeholder alert
            print(f"[ALERT] Break probability {combined:.2f} exceeds threshold!")



