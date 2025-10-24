import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from hmmlearn import hmm
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from statsmodels.tsa.seasonal import STL
import torch
import torch.nn as nn
from scipy.stats import skew
from scipy.stats import entropy

from scipy.stats import kendalltau
from scipy.special import erf
from collections import deque


class Preprocessor:
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



import numpy as np
from math import sqrt, pi, exp

class BOCPDPerspective(Perspective):
    def __init__(self, hazard_lambda: int = 50, runlen_max: int = 200):
        self.hazard_lambda = hazard_lambda
        self.runlen_max = runlen_max
        self.R = np.array([1.0])
        self._sums = [0.0]
        self._sumsqs = [0.0]
        self._counts = [0]

    def _hazard(self, r, shock_score=0.0):
        base = 1.0 / self.hazard_lambda
        adaptive = 0.3 * (1.0 + shock_score)  # stronger if shocks present
        return min(0.5, base + adaptive / (1 + (r if r is not None else 1)))


    def _predictive_prob(self, x, sum_, sumsq, n):
        if n <= 0:
            mu, sigma = 0.0, 1.0
        else:
            mu = sum_ / n
            var = max((sumsq - (sum_**2)/n) / (n - 1) if n > 1 else 1.0, 1e-2)
            sigma = sqrt(var * 2.0)  # inflate variance by factor 2 (try 2–5)

        # Gaussian density, stabilized
        z = (x - mu) / (sigma + 1e-12)
        log_prob = -0.5 * z**2 - np.log(sigma + 1e-12) - 0.5*np.log(2*np.pi)
        return np.exp(log_prob)

    def update(self, x: float, embedding: Optional[np.ndarray] = None, shock_score: float = 0.0) -> float:
        x = float(x)
        R = self.R
        L = min(len(R), self.runlen_max)

        p = np.zeros(len(R))
        for r in range(len(R)):
            sum_, sumsq, n = self._sums[r], self._sumsqs[r], self._counts[r]
            p[r] = self._predictive_prob(x, sum_, sumsq, n)

        H = np.array([self._hazard(r) for r in range(len(R))])
        
        growth = R * p * (1.0 - H)
        cp = np.sum(R * p * H)

        R_new = np.empty(len(growth) + 1)
        R_new[0] = cp
        R_new[1:] = growth

        R_new /= (np.sum(R_new) + 1e-12)
        if len(R_new) > self.runlen_max:
            R_new = R_new[:self.runlen_max]

        new_sums = [x]
        new_sumsqs = [x*x]
        new_counts = [1]
        for r in range(len(self._sums)):
            new_sums.append(self._sums[r] + x)
            new_sumsqs.append(self._sumsqs[r] + x*x)
            new_counts.append(self._counts[r] + 1)

        decay = np.exp(-np.arange(len(R_new)) / 50.0)
        R_new *= decay
        R_new /= np.sum(R_new)

        self.R = R_new[:self.runlen_max]
        self._sums = new_sums[:self.runlen_max]
        self._sumsqs = new_sumsqs[:self.runlen_max]
        self._counts = new_counts[:self.runlen_max]

        # 🔑 don’t let shock_score kill BOCPD
        # ratio-based scoring: R0 vs strongest competitor (ignoring r=0 itself)
        competitor = np.max(self.R[1:]) if len(self.R) > 1 else 1e-12
        ratio = self.R[0] / (self.R[0] + competitor + 1e-12)

        # apply soft downweight for shock influence
        score = ratio * (0.7 + 0.3 * (1.0 - shock_score))


        if np.random.rand() < 0.01:  # occasional print
            print("cp:", cp,
                  "R0:", R_new[0],
                  "maxR:", R_new.max(),
                  "meanR:", R_new.mean())

        
        return float(np.clip(score, 0.0, 1.0))





class ResidualPerspective(Perspective):
    """
    Step 2: Residual-based perspective (embedding-enhanced, refined version)

    Responsibilities:
      - Monitor short-term residual dynamics in an online/univariate series.
      - Capture four 'flavours' of potential structural breaks:
        1. Mean shift
        2. Variance shift
        3. Autocorrelation shift
        4. Embedding drift (from DL preprocessor, local temporal context)
      - Produce a single combined break score in [0,1] per point for the aggregator.

    Notes:
      - Dynamic rolling thresholds to suppress false positives.
      - Conditional shock downweight applied only if other flavours do not corroborate.
      - Optional peak sharpening to emphasize true breaks.
      - EWMA optional for variance stabilization.
      - Scores clamped to [0,1].
      - last_flavours stores individual flavour scores for debugging/plotting.
      - Fully online and sequential.
    """

    def __init__(self, window: int = 20, lag: int = 1, ewma_alpha: float = 0.2,
                 epsilon: float = 1e-8, smooth_alpha: float = 0.05, peak_gamma: float = 1.5):
        """
        Args:
            window (int): Rolling buffer size for residuals and embeddings.
            lag (int): Lag parameter for autocorrelation shift detection.
            ewma_alpha (float): Smoothing parameter for EWMA variance tracking.
            epsilon (float): Small constant for numerical stability.
            smooth_alpha (float): Weight for short-term smoothing of combined score.
            peak_gamma (float): Exponent for optional peak sharpening (>1.0 to boost strong signals)
        """
        self.window = window
        self.lag = lag
        self.ewma_alpha = ewma_alpha
        self.epsilon = epsilon
        self.smooth_alpha = smooth_alpha
        self.peak_gamma = peak_gamma

        self.garch_model = None
  
        self._embed_centroid = None
        self._embed_cov_diag = None
        self._emb_alpha = 0.1

        self._buffer: List[float] = []
        self._embed_buffer: List[np.ndarray] = []
        self._variance_ewma: Optional[float] = None
        self._prev_combined: Optional[float] = None
        self.last_flavours: Dict[str, float] = {}

        
    # -----------------------------
    # Flavour 1: Mean / Level Shift
    # -----------------------------
    # replace z-based score with windowed CUSUM-like standardized change
    def _mean_shift_score(self) -> float:
        buf = np.array(self._buffer[-self.window:])
        if len(buf) < 5:
            return 0.0
        # split into two halves: recent vs baseline
        mid = len(buf)//2
        baseline = buf[:mid]
        recent = buf[mid:]
        # robust mean & scale
        med_b = np.median(baseline)
        mad_b = np.median(np.abs(baseline - med_b)) + self.epsilon
        # CUSUM-like statistic: sum of deviations in recent window scaled by sqrt(n)
        cusum = np.sum(recent - med_b) / (mad_b * np.sqrt(len(recent)))
        score = 1.0 - np.exp(-0.5 * (cusum**2))  # map to (0,1)
        return float(np.clip(score, 0.0, 1.0))


    # -----------------------------
    # Flavour 2: Variance / Vol Shift
    # -----------------------------
    from arch import arch_model

    def _fit_garch(self, buf: np.ndarray):
        """Fit GARCH(1,1) once on baseline buffer."""
        try:
            am = arch_model(buf, vol='GARCH', p=1, q=1, rescale=False)
            res = am.fit(disp="off")
            self._garch_model = res
        except Exception:
            self._garch_model = None

    def _variance_shift_score(self, shock_score: float) -> float:
        buf = np.array(self._buffer[-self.window:])
        if len(buf) < 20:
            return 0.0

        # lazy baseline fit
        if not hasattr(self, "_garch_model") or self._garch_model is None:
            self._fit_garch(buf)
            return 0.0

        try:
            # forecast next-step variance
            fcast = self._garch_model.forecast(horizon=1, reindex=False)
            pred_var = float(fcast.variance.values[-1, 0])
        except Exception:
            return 0.0

        # realized short-window variance
        sw = max(5, len(buf) // 4)
        var_now = np.var(buf[-sw:])
        ratio = var_now / (pred_var + self.epsilon)

        score = min(max(0.0, np.log(1 + abs(ratio - 1.0)) / np.log(2.0)), 1.0)
        return float(score)



    # -----------------------------------
    # Flavour 3: Autocorrelation Dynamics
    # -----------------------------------
    from statsmodels.regression.linear_model import yule_walker

    def _autocorr_shift_score(self):
        buf = np.array(self._buffer[-self.window:], dtype=float)
        if len(buf) < max(10, self.lag+3):
            return 0.0
        # estimate AR(1) coefficients on previous and recent halves
        mid = len(buf)//2
        prev = buf[:mid]
        recent = buf[mid:]
        try:
            phi_prev, _ = yule_walker(prev, order=1)
            phi_now, _ = yule_walker(recent, order=1)
            delta = abs(phi_now[0] - phi_prev[0])
        except Exception:
            # fallback to simple corr
            delta = abs(np.corrcoef(prev[:-1], prev[1:])[0,1] - np.corrcoef(recent[:-1], recent[1:])[0,1])
        score = min(1.0, delta * 3.0)
        return float(score)


    # -----------------------------------
    # Flavour 4: Embedding Drift Dynamics
    # -----------------------------------

    def _embedding_drift_score(self, embedding):
        if embedding is None:
            return 0.0
        emb = np.asarray(embedding, dtype=float)
        if self._embed_centroid is None:
            self._embed_centroid = emb.copy()
            self._embed_cov_diag = np.ones_like(emb) * 1e-6
            return 0.0
        # EWMA update
        self._embed_centroid = self._emb_alpha * emb + (1 - self._emb_alpha) * self._embed_centroid
        diff = emb - self._embed_centroid
        self._embed_cov_diag = self._emb_alpha * (diff**2) + (1 - self._emb_alpha) * self._embed_cov_diag
        # Mahalanobis-ish score
        mscore = np.sqrt(np.sum((diff**2) / (self._embed_cov_diag + self.epsilon)))
        # map to (0,1)
        score = 1.0 - np.exp(-0.5 * (mscore/3.0)**2)
        return float(np.clip(score, 0.0, 1.0))


    # -----------------
    # Main update logic
    # -----------------
    def update(self, x: float, embedding: Optional[np.ndarray] = None, shock_score: float = 0.0) -> float:
        # Update buffers
        self._buffer.append(x)
        if len(self._buffer) > self.window:
            self._buffer.pop(0)

        if embedding is not None:
            self._embed_buffer.append(embedding)
            if len(self._embed_buffer) > self.window:
                self._embed_buffer.pop(0)

        # Compute flavour scores
        mean_shift = self._mean_shift_score()
        variance_shift = self._variance_shift_score(shock_score)
        autocorr_shift = self._autocorr_shift_score()
        embedding_drift = self._embedding_drift_score(embedding)

        # Store last flavours
        self.last_flavours = {
            "mean_shift": mean_shift,
            "variance_shift": variance_shift,
            "autocorr_shift": autocorr_shift,
            "embedding_drift": embedding_drift
        }

        # after computing mean_shift and variance_shift
        # amplify if both are present
        mean_present = mean_shift >= 0.4
        var_present = variance_shift >= 0.4

        # small corroboration multiplier
        if mean_present and var_present:
            # boost both flavour scores to signal strong structural change
            mean_shift = min(1.0, mean_shift * 1.5)
            variance_shift = min(1.0, variance_shift * 1.5)
            # optionally reduce shock downweighting effect on variance_shift
            if shock_score > 0.6:
                variance_shift = variance_shift * (1.0 - 0.25 * shock_score)


        # -----------------------------
        # Conditional Shock Downweight
        # Only downweight variance if other corroborating flavours are weak
        corroboration = (mean_shift > 0.2) or (embedding_drift > 0.15)
        if shock_score > 0.7 and not corroboration:
            variance_shift *= (1.0 - 0.5 * shock_score)

        # Weighted aggregation
        weights = {
            "mean_shift": 1.0,
            "variance_shift": 0.5,
            "autocorr_shift": 0.3,
            "embedding_drift": 0.7
        }
        combined = (
            mean_shift * weights["mean_shift"] +
            variance_shift * weights["variance_shift"] +
            autocorr_shift * weights["autocorr_shift"] +
            embedding_drift * weights["embedding_drift"]
        ) / sum(weights.values())

        # Short-term smoothing (light)
        if self._prev_combined is not None:
            combined = self.smooth_alpha * self._prev_combined + (1 - self.smooth_alpha) * combined
        self._prev_combined = combined

        # Optional peak sharpening to highlight true breaks
        combined = combined ** self.peak_gamma

        # Clamp
        combined = max(0.0, min(1.0, combined))
        return combined


class ThresholdPerspective(Perspective):
    """
    Step 2: Threshold-based perspective (enhanced operational implementation)

    Responsibilities
    ----------------
      - Detect deviations outside rolling thresholds in an online manner.
      - Retains two core thresholding modes:
          * Quantile-based (robust, default).
          * Mean ± k·σ (lighter, less robust).
      - Enhancements:
          * Robust scale (MAD or std).
          * Adaptive thresholding (slow drift allowance).
          * Nonlinear scaling (saturating tail probabilities).
          * Persistence filtering for stability.
          * Optional embedding modulation for contextual adaptation.
          * Shock downweighting for consistency with shock perspective.

    Inputs
    ------
      - x : float
          New observation.
      - embedding : np.ndarray or None
          Optional context vector. Used to modulate thresholds (currently simple).
      - shock_score : float in [0,1]
          Downweights extreme points already flagged as shocks.

    Outputs
    -------
      - score : float in [0,1]
          Contribution to aggregated break probability.

    Parameters
    ----------
      - window : int
          Rolling buffer length.
      - percentile : float in (0,1)
          Tail cutoff probability for quantile mode.
      - k : float
          Std deviation multiplier for mean±kσ mode.
      - use_quantile : bool
          If True, quantile thresholds; if False, mean±kσ.
      - persistence : int
          Minimum consecutive exceedances required before scoring.
      - adapt_rate : float in (0,1)
          Rate of exponential smoothing for adaptive thresholds.
      - nonlinear : bool
          If True, apply saturating nonlinearity (logistic) to raw score.
      - logging : bool
          Enable detailed diagnostics (stored internally).

    Reliability / Mitigations
    -------------------------
      - Early buffer (<5) returns 0.0 for stability.
      - Robust scale (MAD) avoids std collapse in flat data.
      - Persistence prevents isolated spikes from triggering.
      - Nonlinear scaling avoids oversensitivity and ensures [0,1].
      - Adaptive thresholds reduce drift-related false positives.
      - Shock downweight avoids double-counting extreme shocks.
    """

    def __init__(
        self,
        window: int = 50,
        percentile: float = 0.95,
        k: float = 2.0,
        use_quantile: bool = True,
        persistence: int = 1,
        adapt_rate: float = 0.05,
        nonlinear: bool = True,
        logging: bool = False
    ):
        self.window = window
        self.percentile = percentile
        self.k = k
        self.use_quantile = use_quantile
        self.persistence = persistence
        self.adapt_rate = adapt_rate
        self.nonlinear = nonlinear
        self.logging = logging
        self.values = []
        self._persist_count = 0

        # State
        self._buffer: List[float] = []
        self._exceed_count: int = 0
        self._diagnostics: Dict[str, Any] = {}

        # Adaptive thresholds (start unset)
        self._lower_adapt: Optional[float] = None
        self._upper_adapt: Optional[float] = None

        self.epsilon = 1e-8

    def update(self, x: float, embedding: Optional[np.ndarray] = None, shock_score: float = 0.0) -> float:
        # buffer update
        self._buffer.append(x)
        if len(self._buffer) > self.window:
            self._buffer.pop(0)

        if len(self._buffer) < 10:
            return 0.0

        buf = np.array(self._buffer)
        mu, sigma = np.mean(buf), np.std(buf) + self.epsilon

        # soft threshold via Gaussian tail probability
        z = (x - mu) / sigma
        prob_tail = 2 * (1 - 0.5 * (1 + erf(abs(z) / np.sqrt(2))))

        # embedding-driven modulation
        if embedding is not None:
            emb_var = np.var(embedding)
            prob_tail *= 1.0 / (1.0 + emb_var)

        # sliding-window persistence
        self._persist_count = getattr(self, "_persist_count", 0)
        if prob_tail > 0.2:
            self._persist_count += 1
        else:
            self._persist_count = 0

        if self._persist_count >= 3:
            score = min(1.0, prob_tail * 1.2)  # emphasize persistent deviations
        else:
            score = prob_tail

        # shock downweight (soft)
        score *= (0.8 + 0.2 * (1.0 - shock_score))

        return float(np.clip(score, 0.0, 1.0))


    def get_diagnostics(self) -> Dict[str, Any]:
        return self._diagnostics


class FragilityPerspective(Perspective):
    """
    Step 2: Fragility / instability perspective (adaptive, statistically principled)

    Responsibilities:
      - Measure local volatility, autocorrelation growth, and skewness changes.
      - High score indicates fragile, unstable, or rapidly changing dynamics.
      - Aggregates three 'flavours':
        1. Variance growth: local increases in variability.
        2. Autocorrelation growth: lag-1 autocorrelation rising.
        3. Skewness growth: changes in asymmetry of distribution.
      - Produces a single combined break score in [0,1] per point for the aggregator.

    Inputs (per update call):
      - x: scalar float, new observation.
      - embedding: optional np.ndarray, can stabilize/contextualize estimates.
      - shock_score: float in [0,1], downweight extreme points.

    Outputs:
      - combined: float in [0,1], break probability contribution.
      - Internally stores last_flavours dict for optional plotting/debugging.

    Dataflow and logic:
      1. Maintain a rolling buffer of last `window` points.
      2. Each update:
         a. Append new point, drop oldest if buffer exceeds window.
         b. Compute three flavour scores from buffer:
            i. Variance growth (rolling variance of recent subwindow)
            ii. Autocorrelation growth (lag-1, rolling)
            iii. Skewness growth (rolling)
         c. Compute baseline stats (mean/std) per flavour for adaptive weighting.
         d. Combine scores via adaptive weighted average.
         e. Apply robust z-score scaling to [0,1].
         f. Downweight each flavour score by shock_score.
         g. Store flavours internally for diagnostics.
      3. Designed for online point-wise updates.
    """

    def __init__(
        self,
        window: int = 200,
        subwindow: int = 20,
        epsilon: float = 1e-8,
        diagnostic: bool = False,
        mode='gradual'
    ):
        """
        Parameters
        ----------
        window : int
            Total rolling buffer length for online update.
        subwindow : int
            Length of rolling subwindow for local computations (variance, autocorr, skewness).
        epsilon : float
            Small value to avoid divide-by-zero in computations.
        diagnostic : bool
            If True, optionally print debug information for each update.
        """
        self.window = window
        self.subwindow = subwindow
        self.epsilon = epsilon
        self.diagnostic = diagnostic

        self.mode=mode

        self._buffer: List[float] = []
        self.last_flavours: Dict[str, float] = {}

        # Maintain historical flavour stats for adaptive weighting
        self._hist_var_growth: List[float] = []
        self._hist_ac_growth: List[float] = []
        self._hist_skew_growth: List[float] = []

    # --------------------------
    # Flavour score calculations
    # --------------------------
    def _variance_growth_score(self) -> float:
        buf = np.array(self._buffer[-self.subwindow:], dtype=np.float32)
        if len(buf) < 4:
            return 0.0
        mid = max(len(buf)//2, 1)
        var_now = np.var(buf[mid:])
        var_prev = np.var(buf[:mid])
        score = max((var_now - var_prev) / (var_prev + self.epsilon), 0.0)
        return score

    def _autocorr_growth_score(self) -> float:
        buf = np.array(self._buffer[-self.subwindow:], dtype=np.float32)
        if len(buf) < 4:
            return 0.0
        x_t = buf[1:]
        x_tm1 = buf[:-1]
        std_tm1 = np.std(x_tm1)
        std_t = np.std(x_t)
        ac_now = np.corrcoef(x_tm1, x_t)[0,1] if std_tm1 > self.epsilon and std_t > self.epsilon else 0.0

        # baseline: previous lag-1 correlation
        x_prev = buf[:-2]
        y_prev = buf[1:-1]
        std_prev_x = np.std(x_prev)
        std_prev_y = np.std(y_prev)
        ac_prev = np.corrcoef(x_prev, y_prev)[0,1] if std_prev_x > self.epsilon and std_prev_y > self.epsilon else 0.0

        score = max(ac_now - ac_prev, 0.0)
        return score

    def _skewness_growth_score(self) -> float:
        buf = np.array(self._buffer[-self.subwindow:], dtype=np.float32)
        if len(buf) < 4:
            return 0.0
        mid = max(len(buf)//2, 2)
        skew_now = skew(buf[mid:], bias=False)
        skew_prev = skew(buf[:mid], bias=False)
        score = abs(skew_now - skew_prev)
        return score

    # --------------------------
    # Online update logic
    # --------------------------
    
    def update(
        self,
        x: float,
        embedding: Optional[np.ndarray] = None,
        shock_score: float = 0.0
    ) -> float:
        # maintain buffer
        self._buffer.append(x)
        if len(self._buffer) > self.window:
            self._buffer.pop(0)

        recent = np.array(self._buffer[-self.window:])

        # Local detrending
        detrended = recent - np.mean(recent)

        # Variance
        var = np.var(detrended)

        # Autocorrelation (lag-1)
        if len(detrended) > 1:
            ac = np.corrcoef(detrended[:-1], detrended[1:])[0,1]
        else:
            ac = 0

        # Robust skewness
        sk = skew(detrended, bias=False)
        sk = np.sign(sk) * min(abs(sk), 2)  # clip extreme noise

        # Kendall’s tau test for trend in variance
        tau, pval = kendalltau(range(len(detrended)), detrended)
        trend_score = abs(tau) if pval < 0.1 else 0.0

        # Adaptive weighting
        weights = [0.4, 0.3, 0.2, 0.1]  # var, ac, sk, trend
        frag_score = (weights[0]*var + weights[1]*ac + weights[2]*sk + weights[3]*trend_score)

        if self.mode == 'gradual':
            # existing trend-based amplification using Kendall tau etc.
            # (keep previous logic)
            final_score = frag_score  # as computed above
        else:  # 'abrupt' mode: treat sudden var or mean jumps as immediate alerts
            # if subwindow shows sudden var or mean shift, escalate
            if var_growth > 0.4 or abs(detr[-1] - np.median(detr[:-1])) > (2.0 * (np.median(np.abs(detr - np.median(detr))) + self.epsilon)):
                final_score = min(1.0, frag_score * 1.5)
            else:
                final_score = frag_score

        return frag_score



class LatentStatePerspective(Perspective):
    """
    Step 2: Latent-state / regime-based perspective (patched full-fledged HMM implementation)

    Responsibilities:
      - Detect changes in unobserved latent states that manifest as regime shifts.
      - Capture changes in mean, variance, and dynamics indirectly via the latent state sequence.
      - Produce a single break probability score in [0,1] per point for the aggregator.

    Inputs:
      - x: scalar float, new observation.
      - embedding: optional np.ndarray, can help contextualize regime assignment.
      - shock_score: float in [0,1], downweight extreme points.

    Outputs:
      - combined: float in [0,1], break probability contribution.
      - last_states: internal record of posterior probabilities for the most recent point.
      - diagnostics: dictionary with entropy, persistence, likelihood shift, etc.

    Improvements in this patch:
      - Adaptive weighting across sub-metrics to prevent single-metric spikes dominating.
      - EWMA smoothing on posterior differences.
      - Persistence filter for true break emphasis.
      - Dynamic spike threshold based on recent MAD of posterior differences.
      - Shock downweight preserved.
    """
    def __init__(self, n_states=2, window=200):
        self.n_states = n_states
        self.window = window
        self.values = []
        self.hmm = None

    def init_hmm(self, data):
        km = KMeans(n_clusters=self.n_states, n_init=5).fit(data.reshape(-1,1))
        self.hmm = GaussianHMM(n_components=self.n_states, n_iter=100, covariance_type="diag", init_params="mc")
        self.hmm.means_ = km.cluster_centers_
        self.hmm.fit(data.reshape(-1,1))

    def update(self, x, **kwargs):
        self.values.append(x)
        if len(self.values) < self.window:
            return 0.0
        if self.hmm is None:
            self.init_hmm(np.array(self.values[-self.window:]))

        # Likelihood shift
        data = np.array(self.values[-self.window:]).reshape(-1,1)
        logL = self.hmm.score(data)
        avg_logL = logL / len(data)

        # Posterior diff (robust)
        posteriors = self.hmm.predict_proba(data)
        diff = np.mean(np.abs(np.diff(posteriors, axis=0)))
        return abs(avg_logL) * diff


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


class Aggregator:
    """
    Step 3: Aggregates perspective scores using a Linear Opinion Pool (LOP) with
    embedding-driven adaptive weights and special handling for the surprise perspective.

    Responsibilities:
      - Combine scores from all active perspectives into a single break probability.
      - Use Linear Opinion Pool (LOP): weighted average of perspective probabilities.
      - Derive weights dynamically from embeddings through a softmax projection.
      - Treat the surprise (anomaly) perspective as a complementary modulator:
          * Downweights break probability if only surprise flags the point
            and other perspectives disagree.
      - Maintains consistency with score [0,1] invariant.

    Inputs:
      - scores: Dict[str, float], mapping perspective names to latest scores.
        Example keys: 'residual', 'threshold', 'fragility', 'latent_state', 'surprise'
      - embedding: np.ndarray, mandatory input used to compute perspective weights.

    Outputs:
      - combined: float in [0,1], overall break probability for current point.

    Dataflow and logic:
      1. Separate the "surprise" perspective from others.
      2. Project embedding into per-perspective weights via softmax.
      3. Combine scores linearly: posterior = Σ w_i * score_i.
      4. Apply surprise modulation:
          a. If surprise >> other scores, downweight combined posterior.
          b. Surprise alone cannot trigger high break probability.
      5. Clamp result to [0,1].
      6. Return combined float.

    Why Linear Opinion Pool?
      - Unlike Logarithmic Opinion Pool, LOP avoids "collapse to zero"
        when perspectives disagree or some scores are low.
      - More robust in domain-agnostic settings with multiple noisy perspectives.
      - Embedding-driven weights make the pool adaptive to context.

    Example usage:
        aggregator = Aggregator()
        scores = {
            'residual': 0.2,
            'threshold': 0.1,
            'fragility': 0.05,
            'latent_state': 0.15,
            'surprise': 0.9
        }
        embedding = np.random.randn(16)
        combined = aggregator.aggregate(scores, embedding=embedding)
    """

    def __init__(self, diagnostic_logging: bool = False):
        """
        Initialize aggregator state.

        Parameters
        ----------
        diagnostic_logging : bool
            If True, prints diagnostic information about aggregation.
        """
        self._posterior_state: Optional[float] = None
        self._diagnostic_logging = diagnostic_logging

        # Internal weight projection parameters for embeddings
        self._weight_matrix: Optional[np.ndarray] = None
        self._perspective_names: List[str] = []

        self.alpha = 1.0  # Dirichlet prior for weights

    def calibrate_from_synthetic(self, emb_dim: int = 16, n_trials: int = 200, noise_scale: float = 1.0):
        """
        Unsupervised calibration: generate synthetic scenarios and set:
          - self.alpha (Dirichlet prior)
          - self._temp (softmax temperature)
          - self._dominant_threshold and _dominant_boost_factor
        This is heuristic but helps avoid ad-hoc manual tuning.
        """
        rng = np.random.default_rng(seed=123)
        # generate synthetic embeddings and synthetic perspective score vectors
        temps = []
        dominants = []
        alphas = []
        for _ in range(n_trials):
            # generate random embedding
            emb = rng.normal(size=(emb_dim,))
            # create a scenario: one perspective dominant or balanced
            if rng.random() < 0.5:
                # dominant scenario
                scores = rng.random(5) * 0.2
                idx = rng.integers(0, len(scores))
                scores[idx] = 0.9 + 0.1 * rng.random()
            else:
                # balanced high scenario
                scores = rng.random(5) * 0.6 + 0.2
            # compute weight logits using current weight matrix if available else random
            if getattr(self, "_weight_matrix", None) is None:
                logits = rng.normal(scale=0.1, size=(emb_dim, len(scores))).mean(axis=0)
            else:
                logits = emb @ self._weight_matrix
            # measure spread of logits -> inverse temperature estimate
            spread = np.std(logits)
            temps.append(max(0.3, min(1.5, 1.0 / (spread + 1e-6))))
            # dominant heuristics
            dominants.append(np.max(scores))
            alphas.append(1e-3)

        # set hyperparams as robust medians
        self._temp = float(np.median(temps))
        self._dominant_threshold = float(max(0.7, min(0.95, np.median(dominants))))
        self._dominant_boost_factor = 0.9
        self.alpha = float(np.median(alphas))
        # diagnostic print
        if self._diagnostic_logging:
            print(f"[Aggregator.calibrate] temp={self._temp:.3f}, dom_th={self._dominant_threshold:.3f}, alpha={self.alpha:.6f}")


    def _compute_weights(self, perspective_names: List[str], embedding: np.ndarray) -> Dict[str, float]:
        """
        Compute adaptive weights for each perspective using softmax projection of the embedding.

        Parameters
        ----------
        perspective_names : List[str]
            List of perspective names excluding 'surprise'.
        embedding : np.ndarray
            Dense feature vector from DL preprocessor.

        Returns
        -------
        Dict[str, float]
            Mapping from perspective name to adaptive weight (sums to 1).
        """
        dim = len(embedding)
        n_persp = len(perspective_names)

        # Initialize weight matrix lazily
        if self._weight_matrix is None or len(self._perspective_names) != n_persp:
            rng = np.random.default_rng(seed=42)
            self._weight_matrix = rng.normal(0, 0.1, size=(dim, n_persp))
            self._perspective_names = perspective_names

        logits = embedding @ self._weight_matrix  # shape (n_persp,)
        weights = np.exp(logits - np.max(logits))  # softmax numerically stable
        weights /= np.sum(weights)

        return {name: w for name, w in zip(perspective_names, weights)}

    def aggregate(self, scores: Dict[str, float], embedding: Optional[np.ndarray] = None, shock_score: float = 0.0) -> float:
        """
        Aggregate perspective scores into a single break probability.

        - Uses a Dirichlet-like unsupervised weighting of the raw perspective scores.
        - Embedding modulates the weights multiplicatively.
        - Conditional shock downweight: only downweights when shock is present but
          there is insufficient corroboration among core perspectives.
        - Dominant-perspective boost: if any core perspective is extremely confident,
          allow it to push the combined score up (prevents averaging away a real strong signal).
        """
        # Separate surprise and (optional) shock keys from core perspectives
        surprise_score = float(scores.get("surprise", 0.0))
        # do not include 'surprise' or 'shock' in core_scores
        core_scores = {k: float(v) for k, v in scores.items() if k not in ("surprise", "shock")}

        # If no core perspectives, return 0
        if not core_scores:
            return 0.0

        # Build vals vector (floats)
        vals = np.array(list(core_scores.values()), dtype=float)

        # Quick guard: if all zeros
        if np.sum(vals) == 0.0:
            return 0.0

        # 1) Unsupervised / Bayesian-style Dirichlet weighting (keeps your original spirit)
        # self.alpha should exist; default idea: small positive prior
        alpha = getattr(self, "alpha", 1e-3)
        weights = (vals + alpha) / (np.sum(vals) + len(vals) * alpha)  # shape (n_persp,)

        # 2) Embedding-conditioned multiplicative modulation (optional)
        if embedding is not None:
            emb_norm = float(np.linalg.norm(embedding))
            # multiplicative modulation in (1, 1 + something) so relative order preserved
            emb_mod = 1.0 + np.tanh(emb_norm) / (1.0 + emb_norm)
            weights = weights * emb_mod

        # 3) Normalize weights
        weights = weights / (np.sum(weights) + 1e-12)

        # 4) Compute LOP combined score
        combined = float(np.dot(weights, vals))

        # 5) Conditional shock downweighting
        #    Only downweight if shock_score is high AND there is NOT corroboration among core perspectives.
        #    Corroboration threshold: at least 2 core perspectives >= corroboration_threshold
        corroboration_threshold = getattr(self, "_corroboration_threshold", 0.6)
        required_corroborators = getattr(self, "_required_corroborators", 2)

        strong_count = sum(1 for v in core_scores.values() if v >= corroboration_threshold)

        if shock_score is None:
            shock_score = 0.0
        shock_score = float(shock_score)

        if (shock_score > 0.7) and (strong_count < required_corroborators):
            # original soft downweight (you used factor 0.5 earlier); preserve but only when uncorroborated
            combined *= (1.0 - 0.5 * shock_score)

        # 6) Surprise modulation (preserve previous 'surprise' logic as a soft veto)
        if surprise_score > 0.7 and combined < 0.2:
            combined *= (1.0 - 0.5 * surprise_score)

        # 7) Dominant-perspective boost: if any core perspective is very confident,
        #    let that perspective largely determine the combined score.
        #    This prevents the LOP averaging a single strong signal away.
        max_name, max_score = max(core_scores.items(), key=lambda kv: kv[1])
        dominant_threshold = getattr(self, "_dominant_threshold", 0.85)
        dominant_boost_factor = getattr(self, "_dominant_boost_factor", 0.9)

        if max_score >= dominant_threshold:
            combined = max(combined, dominant_boost_factor * max_score)

        # 8) Final clamp + diagnostic logging + store posterior state
        combined = float(np.clip(combined, 0.0, 1.0))
        self._posterior_state = combined

        if self._diagnostic_logging:
            # report core_scores and computed weights alongside surprise and shock
            try:
                weight_map = {name: float(w) for name, w in zip(core_scores.keys(), weights)}
            except Exception:
                weight_map = {}
            print(f"[Aggregator] core_scores={core_scores}, weights={weight_map}, "
                  f"surprise={surprise_score:.3f}, shock={shock_score:.3f}, combined={combined:.3f}")

        return combined



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



# -----------------------------
# MAIN LOOP / ONLINE PROCESSING
# -----------------------------
# Responsibilities:
#   - Load series, optionally select random series for demonstration.
#   - Sequentially apply preprocessing, DL preprocessing, shock detection.
#   - Update each perspective online with latest point and embedding.
#   - Aggregate perspective scores to combined break probability.
#   - Post-process combined scores via smoothing/hysteresis.
#   - Record outputs and optionally issue alerts.
#   - Plot final series with perspective scores and combined probability.
#
# Dataflow:
#   1. Preprocess series using classical Preprocessor.
#   2. For each point:
#       a. Apply DLPreprocessor -> cleaned point + embedding.
#       b. Compute shock score -> downweight extreme points.
#       c. Update each perspective -> get float score in [0,1].
#       d. Aggregate perspective scores -> combined break probability.
#       e. Post-process combined scores for smoothing.
#       f. Record in OutputHandler.
#   3. Plot all perspective scores and combined score for visualization.


# -------------------------------
# MAIN PIPELINE TEST RUN
# -------------------------------
if __name__ == "__main__":

    # -------------------------------
    # Load example series
    # -------------------------------
    df = pd.read_csv("f:/synth_ts.csv")
    series_ids = df['series_id'].unique()
    random_id = "ecology_set1" #np.random.choice(series_ids)
    random_series = df[df['series_id'] == random_id].sort_values("time")
    ts = pd.Series(data=random_series['value'].values, index=random_series['time'])
    print(f"Randomly selected series: {random_id}")
    print(ts.head())

    # -------------------------------
    # Initialize modules
    # -------------------------------
    preprocessor = Preprocessor()
    dlpreprocessor = DLPreprocessor()
    shock_detector = ShockDetector()
    bocpd = BOCPDPerspective()
    residual = ResidualPerspective()
    threshold = ThresholdPerspective()
    fragility = FragilityPerspective()
    latent = LatentStatePerspective()
    surprise = SurprisePerspective()
    aggregator = Aggregator()
    output_handler = OutputHandler()
    postprocessor = PostProcessor()

    # Dictionary of perspectives for easy iteration
    perspectives = {
        'bocpd': bocpd,
        'residual': residual,
        'threshold': threshold,
        'fragility': fragility,
        'latent': latent,
        'surprise': surprise
    }

    # -------------------------------
    # Preprocess series
    # -------------------------------
    ts_preprocessed = preprocessor.preprocess(ts)

    # -------------------------------
    # Prepare storage for scores
    # -------------------------------
    all_scores = {name: [] for name in perspectives}
    combined_scores = []
    shock_scores = []

    # -------------------------------
    # Run pipeline online
    # -------------------------------
    for t, x in ts_preprocessed.items():
        # DL preprocessing: cleaned point + embedding
        x_clean, emb = dlpreprocessor.transform(x)

        # Shock detection: downweight extreme points
        shock_score = shock_detector.update(x_clean)
        shock_scores.append(shock_score) 

        # Update perspectives
        scores = {}
        for name, perspective in perspectives.items():
            scores[name] = perspective.update(x_clean, embedding=emb, shock_score=shock_score)
            all_scores[name].append(scores[name])

        # Aggregate perspective scores into combined break probability
        aggregator.calibrate_from_synthetic(emb_dim=len(emb))
        combined = aggregator.aggregate(scores, embedding=emb)
        combined_scores.append(combined)

        # Post-process combined score (smoothing)
        smoothed = postprocessor.smooth(combined_scores)
        alert_flag = smoothed[-1] >= output_handler.alert_threshold

        # Record outputs
        output_handler.record(scores, combined)


    # -------------------------------
    # Plot results
    # -------------------------------
    plt.figure(figsize=(14,7))
    plt.plot(ts_preprocessed.index, ts_preprocessed.values, label="Preprocessed series", color='black')
    for name, scores_list in all_scores.items():
        plt.plot(ts_preprocessed.index, scores_list, label=f"{name} perspective score")
    plt.plot(ts_preprocessed.index, combined_scores, label="Combined score", linestyle='--', color='red')
    plt.plot(ts_preprocessed.index, shock_scores, label="Shock detector score", linestyle='-.', color='orange')
    plt.axhline(y=1.0, color='blue', linestyle=':', label='y = 1.0')
    plt.xlabel("Time")
    plt.ylabel("Value / Score")
    plt.legend()
    plt.title(f"Series {random_id} - All perspectives + combined")
    plt.show()
