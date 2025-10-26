This is a crude, preliminary attempt to bring together a number of different methods of changepoint detection in an integrated fashion.  
**Note:** This is not a working version and does not produce the expected results. This is only to archive the methodological approach.

This is an experiment and exploration of the methodology and as such is a mix of vibe coding and manual implementation.

We began with a set of high-level questions that guided our dive:

• Definition and scope of changepoints/structural breaks:  
→ “What is a changepoint really, across fields?”

• Complications in detection:  
→ “What about trends, seasonality, mean reversion — do they distort detection? When does it matter (diagnostic vs. forecasting)?”

• Gradual vs. abrupt changes:  
→ “If change isn’t instantaneous but unfolds over timesteps, how is that treated? Is it still a ‘break’?”

• Framings / structuring of detection:  
→ “What are the main families of methods, how do they ‘think’ about change? Sliding windows vs. global fits vs. sequential monitoring?”

• Distortions and confounds:  
→ “What else could look like a break but isn’t (nonlinearities, data artifacts, local changes, etc.)?”

• Preprocessing, reclocking, unit of observation:  
→ “What if we redefine the time unit or aggregation? Can we detect breaks better if we change how the clock ticks?”

• Domain-agnostic vs. domain-specific validity:  
→ “How reliable are reclocking or preprocessing tricks if I don’t know the data’s context (finance vs. seismology vs. biology)?”

---

We expand into a multi-dimensional view of changepoint detection:

**(A) Conceptual foundations**  
• Structural break = usually treated as point-like, but may be gradual.  
• Different fields emphasize different analogies: finance (policy regime), climate (trend shifts), biology (mutation events).

**(B) Distortions / confounds in detection**  
Seven categories we mapped out:  
1. Trends / drifts – long-run shifts can masquerade as breaks.  
2. Seasonality / cyclicity – periodic changes mistaken as regime change.  
3. Mean reversion / persistence – temporary shocks may be over-read as structural.  
4. Nonlinearities / thresholds – hidden regime-switch dynamics produce break-like patterns.  
5. Data issues / artifacts – sensor errors, reporting lags, market closures.  
6. Multi-scale structure – micro vs. macro breaks.  
7. Gradual vs. abrupt change – fuzzy boundaries between break and trend.

**(C) Structuring detection families**  
Three overarching families:  
1. Window-based / local comparisons – sliding tests of before vs. after.  
2. Global model-fitting – segmenting whole series via likelihood or penalties.  
3. Sequential monitoring – online detection, CUSUM/SPRT type.  
Each has different strengths and vulnerabilities to distortions.

**(D) Gradual change vs. trend**  
• Distinguishing “slow break” from “trend” depends on assumptions:  
• Break = regime with a new stationary state after change.  
• Trend = continuous evolution, no return to stability.  
• Possible strategies:  
  - Fit trend models + test residuals for breakpoints.  
  - Multi-resolution detection (look at different scales).  
  - Hybrid models (trend + changepoint in slope).

**(E) Reclocking & unit change**  
• Beyond rescaling: redefining the “clock” itself (event-time, volatility-time, CUSUM transforms).  
• Domain-agnostic: generic statistical reclocking (cumulative sums, block means).  
• Domain-specific: semantic clocks (volume bars in markets, stress release in seismology).  
• Concern: reliability if domain is unknown → safest to stay with purely statistical transformations.

---

To deepen our search, here are related dimensions worth considering:

1. Evaluation perspective  
• Diagnostic vs. predictive uses: when is it enough to mark a break, and when do you need to forecast post-break dynamics?

2. Model robustness  
• How do methods behave under misspecification (e.g. heavy tails, nonlinear DGPs)?  
• How sensitive are they to window size, penalty choice, or tuning parameters?

3. Multi-resolution & hierarchical detection  
• Many systems exhibit nested breaks (small/local vs. large/global).  
• How to design methods that catch both without confusing one for the other?

4. Uncertainty quantification  
• Not just “where is the break” but “with what confidence?”  
• Especially relevant for forecasting tasks.

5. Synthetic vs. empirical testing  
• Benchmark on artificial series (controlled distortions) vs. messy real data (domain-unknown).

6. Cross-disciplinary analogies  
• Finance (policy shift, liquidity regime).  
• Climate (abrupt shift in mean temperature).  
• Genomics (copy number variation).  
• Epidemiology (disease spread acceleration).  
• Each field has unique ways to reconcile gradual vs abrupt, trend vs regime.

---

Here's a structured view of the method families to cover all grounds:

| Perspective                             | Core Intuition / Narrative                                                                 | How It Differs from Fitted-Model Perspective                                              | Treatment of DGP / Implication                                                                 |
|----------------------------------------|---------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| Regime / State Shift                   | System exists in multiple states; break occurs when the system moves to a new state         | Doesn’t rely on past model fit; focuses on a qualitative change in state                  | DGP is piecewise: each regime has its own generating process; break = transition between DGPs |
| Force-Driven / Causal Change           | Real-world forces (inputs, policies, shocks) change                                         | Focus is on causes, not on model fit; break is evidence of external force change          | DGP parameters are altered by exogenous or endogenous forces; break reflects new causal structure |
| Threshold / Boundary-Crossing         | System behaves continuously until a critical point; break occurs after crossing             | Fitted-model may not anticipate non-linear thresholds; break is about system non-linearity | DGP is non-linear; break emerges from reaching a critical point, not misfit of linear model    |
| Surprise / Anomaly                     | A disruptive event changes system behavior                                                  | Focus on unexpected events, not gradual model misfit                                      | DGP changes abruptly due to rare or exogenous shocks; may create transient or permanent regime change |
| Process / Mechanism Change            | The process generating data itself changes                                                  | Doesn’t rely on past model fitting; break is change in rules/mechanisms                   | DGP itself is altered; new process replaces or modifies old one; not just parameter shift      |
| Information / Knowledge Update        | Break occurs when agents’ understanding of the system changes                               | Break is driven by perception, not past fit                                               | DGP may be the same objectively, but effective DGP (observed outcomes) shifts because behavior or expectations change |
| Interaction / Network Change          | Break arises from changes in relationships among system components                          | Break is systemic, not single-variable misfit                                             | DGP becomes higher-order: correlations, dependencies, or network effects change, generating new emergent behavior |
| Structural Fragility / Instability    | Small shocks trigger large changes due to system’s instability                              | Fitted-model may not capture tipping points; break is endogenous                          | DGP is sensitive/non-linear; break is amplification of latent vulnerabilities, not misfit per se |
| Regime-Contingent Expectations / Behavioral Change | Break occurs because collective expectations/behavior change                        | Break is about agent coordination, not past statistical fit                               | DGP changes as a result of endogenous behavior adaptation, not purely from external shocks     |


Then refined to arrive at this:

| Perspective / Flavour                         | Core Idea                                                                 | Conceptual Distinction from “generic fitted-model”                                | Quantitative Approach (Univariate)                                               | Type of DGP Change Detected                                      |
|----------------------------------------------|---------------------------------------------------------------------------|------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|------------------------------------------------------------------|
| Fitted-model: mean/level-based               | Break occurs when mean level of series shifts                             | Focuses only on level changes                                                     | CUSUM, Chow test, rolling mean comparison                                        | Shift in mean of DGP                                             |
| Fitted-model: variance-based                 | Break occurs when variability changes                                     | Focuses on second moment, not level                                               | Rolling variance, GARCH parameter shifts, CUSUM of squares                       | Shift in variance/volatility of DGP                              |
| Fitted-model: autocorrelation / dynamics-based | Break occurs when temporal dependence changes                             | Focuses on autocorrelation structure or ARMA parameters                           | Recursive estimation of ARMA coefficients, residual diagnostics                  | Change in temporal dependence structure of DGP                   |
| Threshold / boundary-crossing                | Break occurs when series crosses a critical value triggering new dynamics | Not defined by past fit; break triggered by hitting threshold                     | Threshold autoregressive (TAR) models, piecewise regression, change-point in level crossings | Non-linear DGP shifts triggered by observed value                |
| Structural fragility / instability           | Break arises from internal amplification of small shocks                  | Focus on emergent instability rather than fit misalignment                        | Early warning indicators: variance, autocorrelation, skewness over rolling windows | DGP exhibits critical transitions or tipping points              |
| Regime / behavioral-like (Markov-switching univariate) | Series shifts between latent states with different properties     | Detects shifts in latent structure, not just residual misfit                      | Univariate Markov-switching models, hidden Markov models                         | DGP switches between latent regimes with different parameters    |
| Interaction/network perspective              | Not applicable in univariate series                                       | Requires multivariate data; cannot operate on single series                       | N/A                                                                              | N/A                                                              |
| Force-driven / causal perspective            | Not applicable in univariate series without exogenous input               | Needs external driver information                                                 | N/A                                                                              | N/A                                                              |
| Surprise / anomaly perspective               | Could be implemented as extreme deviations from prior distribution        | Uses probabilistic surprise rather than model misfit                              | Outlier detection, extreme value analysis, sequential probability ratio test     | Rare or extreme events in DGP, may coincide with level/variance shifts |

---

And eventually to this — to be univariate, online and domain-agnostic:

| Perspective / Flavour                         | Core Idea                                                                 | Conceptual Distinction from Generic Fitted-Model                                  | Key Focus / What It Looks For                          | Conceptual Diagnostic Approach                                                                 | Issues / Caveats                                                                 | Checks / Measures for Reliability                                                                 |
|----------------------------------------------|---------------------------------------------------------------------------|------------------------------------------------------------------------------------|--------------------------------------------------------|-----------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| Fitted-model: mean / level                   | Break occurs when the series’ average shifts                              | Focuses on persistent deviation from historical mean; residual-based              | Change in mean                                         | Examine residuals over time; persistent nonzero mean indicates a shift                          | Misspecification of baseline mean; gradual changes may be hard to detect         | Confidence intervals on mean, rolling z-scores, Bayesian credible intervals for mean estimate     |
| Fitted-model: variance / volatility          | Break occurs when variability changes                                     | Focuses on second moment rather than level; residual-based                        | Change in variance                                     | Monitor residual heteroscedasticity or rolling variance; sustained change indicates break       | Volatility clustering may mimic a break; extreme outliers may distort variance   | Rolling variance confidence intervals, t- or F-statistics for variance comparison, bootstrap variance estimates |
| Fitted-model: autocorrelation / dynamics     | Break occurs when temporal dependence structure changes                   | Looks at lag correlations or ARMA parameters; residual-based                      | Change in autocorrelation or lagged relationships      | Examine residual autocorrelation over rolling windows; systematic changes indicate break        | Small sample sizes may bias autocorrelation; slow gradual shifts may be missed   | Confidence intervals on autocorrelations, rolling AR parameter intervals, Ljung-Box type diagnostics |
| Threshold / boundary-crossing                | Break occurs when series crosses critical values triggering new dynamics  | Not defined by past model fit; break triggered by hitting threshold               | Observed values relative to critical thresholds        | Identify points where series exceeds/falls below threshold and exhibits different behavior      | Choice of threshold can be arbitrary; slow approaches may be ambiguous           | Probabilistic thresholds (historical percentiles), rolling z- or t-scores relative to mean/variance, sensitivity analysis to threshold |
| Structural fragility / instability           | System becomes sensitive; small shocks are amplified                      | Focus on latent system sensitivity rather than residual misfit                    | Early-warning signs: rising variance, autocorrelation, skewness | Track variance, autocorrelation, skewness over rolling windows; trends signal fragility         | Small sample noise may mimic instability; false positives are possible           | Rolling window confidence intervals, bootstrapped estimates, comparison with stochastic null models |
| Regime / latent-state (Markov-switching)     | Series shifts between unobserved states with distinct properties          | Breaks due to latent state change, not residual misfit                            | Periods with distinct mean, variance, or autocorrelation | Assess whether series exhibits clusters of statistically distinct behavior over time            | Number of states may be misestimated; short regimes may be hard to detect         | Posterior probabilities of states, likelihood-based confidence, sensitivity to assumed number of states |
| Surprise / anomaly                           | Break occurs via rare or extreme deviations                               | Focus on extreme events rather than persistent misfit                             | Observations far from historical patterns              | Identify unusually large deviations from past distribution; extreme observations signal break   | Extreme outliers may be noise; multiple comparisons may inflate false positives   | Z- or t-scores for extreme values, probability of exceedance, rolling quantiles, bootstrap of extreme statistics |

This also should consider global vs. local and model-free vs. model-based approaches (so we deal with fitting which raises complexity and risk of errors).

---

We summarise our approach as:

### Step 0: Input & Preprocessing

• Input: raw univariate series (sequential, online)  
• Preprocessing tasks:  
  - Detrending / normalization: remove long-term trends or normalize scale to avoid spurious detections. Methods may include rolling z-scores or differencing.  
  - De-seasonalization / cyclicality removal: identify and remove strong periodic components if present.  
  - Smoothing / noise reduction (optional): lightweight filters to reduce high-frequency noise without losing break points.  
• Output: cleaned / normalized series ready for analysis.

---

### Step 1: Outlier / Shock Handling

• Objective: detect extreme single-step deviations that may bias break detection  
• Design:  
  - Detect shocks via rolling statistics (e.g., deviation from rolling median / variance)  
  - Flag separately or feed into perspectives as part of “surprise/anomaly” branch  
• Scoring: probabilistic score = deviation magnitude relative to rolling or historical variance

---

### Step 2: Perspective-Specific Detection Modules

• Objective: multiple complementary “views” on potential breaks  
• Modules:

| Perspective Module            | Conceptual Approach                                                                 | Notes                                                                 |
|------------------------------|--------------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| Residual (Mean/Variance/Autocorr) | Fit short-term predictive models (rolling mean, rolling variance, AR for autocorrelation). Breaks flagged if residuals deviate beyond expectation. | Captures level, volatility, and dependence structure changes.         |
| Threshold                    | Monitor distribution tails; detect unusual extremes relative to empirical distribution. | Simple, robust; aligns with anomaly detection methods.                |
| Fragility                    | Monitor stability of fitted model parameters; detect when model fit degrades.         | Approximates concept of system near a tipping point.                  |
| Latent-State                 | Sequential latent models (HMM, online clustering, change-point priors). Detect if the system is better explained by a new hidden state. | More complex; allows regime switches.                                 |
| Surprise / Anomaly           | Pointwise anomaly detection via rolling z-score, percentile ranks, or predictive uncertainty. | Quick, online, aligns with intuitive "unexpectedness."                |

• Outputs: each perspective produces:  
  - Probability of break ∈ [0,1], where 1 = break  
  - (Optional) confidence/reliability score of its own detection



| Perspective / Flavour              | Core Idea                                                       | Conceptual Distinction from Generic Fitted-Model                          | Key Focus / What It Looks For                  | Conceptual Diagnostic Approach                                                                 | Issues / Caveats                                                                 | Checks / Measures for Reliability                                                                 |
|-----------------------------------|------------------------------------------------------------------|----------------------------------------------------------------------------|------------------------------------------------|-----------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| Fitted-model: mean / level        | Break occurs when the series’ average shifts                     | Focuses on persistent deviation from historical mean; directly checks residuals | Change in mean                                 | Examine residuals over time; persistent nonzero mean indicates a shift                          | Misspecification of baseline mean; gradual change may be hard to detect         | Confidence intervals on mean, rolling z-scores, Bayesian credible intervals for mean estimate     |
| Fitted-model: variance / volatility | Break occurs when variability changes                             | Focuses on second moment rather than level                                | Change in variance                              | Monitor residual heteroscedasticity or rolling variance; sustained change indicates break       | Volatility clustering may be mistaken for a break; extreme outliers may distort variance | Rolling variance confidence intervals, t- or F-statistics for variance comparison, bootstrapped variance estimates |
| Fitted-model: autocorrelation / dynamics | Break occurs when temporal dependence structure changes           | Looks at lag correlations or ARMA parameters; still residual-based        | Change in autocorrelation or lagged relationships | Examine residual autocorrelation over rolling windows; systematic changes indicate break        | Small sample sizes may bias autocorrelation; slow gradual shifts may be missed   | Confidence intervals on autocorrelations, Ljung-Box type statistics, rolling AR parameter intervals |
| Threshold / boundary-crossing     | Break occurs when series crosses critical thresholds that trigger new behavior | Not defined by past model fit; break is triggered by hitting threshold     | Observed values relative to critical thresholds | Identify points where series exceeds/falls below threshold and exhibits different behavior      | Choice of threshold can be arbitrary; slow threshold approaches may be ambiguous | Probabilistic thresholds using historical percentiles, z- or t-scores relative to mean and variance, sensitivity analysis to threshold |
| Structural fragility / instability | System becomes sensitive; small shocks lead to large changes      | Focus on emergent system sensitivity rather than residual misfit          | Early signs of latent instability: rising variance, autocorrelation, skewness | Track variance, autocorrelation, skewness over rolling windows; detect trends signaling fragility | Small sample noise may mimic instability; early-warning signals can be false positives | Rolling window confidence intervals, bootstrap estimates, comparison with null stochastic simulations |
| Regime / latent-state             | Series switches between unobserved states with distinct properties | Breaks are due to shifts in latent states, not just residual misfit        | —                                              | —                                                                                             | —                                                                                | —                                                                                                 |

---

### Step 3: Score Normalization & Aggregation

• Objective: unify signals from perspectives into one probability per point  
• Options:  
  - Weighted sum / average (weights = reliability scores)  
  - Maximum score (if any perspective strongly signals a break)  
  - Bayesian pooling / probabilistic integration (preferred default for rigor)  
• Output: single break probability ∈ [0,1]

---

### Step 4: Output

• For each new point:  
  - Final probability of break  
  - Perspective-specific scores (for transparency / explainability)  
  - Optional alert if probability exceeds threshold

---

### Step 5: Post-Processing / Optional Smoothing

• Objective: stabilize outputs in noisy environments  
• Options:  
  - Rolling smoothing or exponential smoothing of aggregated probability  
  - Hysteresis / persistence rules (e.g., require N consecutive high-prob signals)  
  - Adaptive thresholding to control false alarms

---

## Implementation Guidelines

### 1. Architecture & Structure

• All implementations must follow the Reference Plan steps (0–5)  
• Every step is modularized (class/function)  
• Order and presence of modules must be preserved (e.g., preprocessing → shock detection → perspectives → aggregation → output → post-processing)  
• Skeleton code is authoritative: new code extends or refines it, not rewrite or bypass it

---

### 2. Input/Output Rules

• Input: raw univariate sequential data (online first, batch second)  
• Intermediate outputs:  
  - Preprocessed series  
  - Shock detector flags/scores  
  - Perspective-specific probabilities  
• Final outputs:  
  - Break probability ∈ [0,1]  
  - Optional perspective breakdown  
  - Optional alerts if threshold exceeded  
• All outputs must remain probabilistic, not hard binary unless post-processed

---

### 3. Preprocessing (Step 0)

• Always include:  
  - Detrending / normalization  
  - De-seasonalization / cycle removal  
  - Optional smoothing  
• Methods must be lightweight, online-compatible (rolling stats, online filters)  
• Must not “over-clean” to the point of erasing actual breaks

---

### 4. Shock Detection (Step 1)

• Implemented immediately after preprocessing  
• Detect extreme single-step deviations (candidate shocks)  
• Score shocks relative to rolling variance or historical scale  
• Must integrate into:  
  - Surprise/anomaly perspective (if treated as anomalies)  
  - Or flagged separately (metadata)

---

### 5. Perspective Modules (Step 2)

• Each perspective is its own module/class, outputs probability ∈ [0,1]  
• Must preserve the five canonical perspectives:  
  1. Residual (mean, variance, autocorrelation changes)  
  2. Threshold  
  3. Fragility  
  4. Latent-state  
  5. Surprise/anomaly  
• Residual-based methods require rolling short-term models (AR, variance, etc.)  
• Latent-state approximated via lightweight online HMM or clustering (no heavy offline training unless explicitly allowed)  
• Fragility proxies: monitor stability/degradation of model fits  
• All must produce probability in [0,1], not raw stats

---

### 6. Aggregation (Step 3)

• All perspective scores must be combined into a single break probability  
• Methods allowed:  
  - Weighted sum/average (weights = reliability)  
  - Maximum rule  
  - Bayesian pooling (default where possible)  
• Aggregator must accept perspective reliability/confidence scores when available

---

### 7. Post-Processing (Step 5)

• Optional but always supported  
• Methods allowed:  
  - Rolling/exponential smoothing  
  - Hysteresis (require persistence before switching states)  
  - Adaptive thresholding  
• Must never overwrite raw probabilities — smoothing is additional output

---

### 8. Coding Standards

• Consistency with skeleton: all functions, classes, and steps exist in code  
• Comments:  
  - Each function/class has docstring: purpose, inputs, outputs, assumptions  
  - Inline comments explain reasoning and tie back to Reference Plan  
  - Must explicitly state if approximation is used instead of full method  
• Naming:  
  - Explicit, consistent (e.g., ShockDetector, ResidualPerspective, Aggregator)  
• Testing hooks:  
  - Each step returns intermediate outputs for debugging

---

### 9. DL / ML Readiness

• Code must remain lightweight and modular, but DL/ML components can be plugged in later  
• No hard-coded dependencies on DL now  
• In Step 2 and Step 3, ensure easy swap-in of:  
  - Neural nets for perspectives  
  - Learned weights for aggregation

---

### 10. Invariants

• Probabilistic outputs everywhere (0 = no break, 1 = certain break)  
• No step skipping: even trivial implementations must preserve the structure  
• Explainability preserved: perspective-specific scores and shock flags available downstream  
• Online-first: all methods default to sequential updating; batch is special case  
• Code additions/patches must not silently break skeleton compliance

---

### 11. Meta Rules

• Optimizations (Numba, vectorization, pruning DL, etc.) can be applied, but must not alter logical flow




Our initial Python skeleton is:

---

Locked-in skeleton for online univariate structural-break detection.

**Structure:**
- Step 0: Preprocessing → `Preprocessor` / `DLPreprocessor` (cleaned point, embedding)  
- Step 1: Shock detection → `ShockDetector` (shock_score)  
- Step 2: Perspectives → `Residual` / `Threshold` / `Fragility` / `Latent` / `Surprise`  
- Step 3: Calibration → `DLCalibrationHelper` (calibrate scores)  
- Step 4: Aggregation → `Aggregator.pool(scores_vector, weights=None)`  
- Step 5: Postprocessing → `PostProcessor` (smoothing/hysteresis)  
- Output → `OutputModule.format/store`

**Guiding invariants (enforced by this skeleton):**
- All perspective batch outputs are DataFrames with column `'combined'` in [0,1]  
- All perspective `update()` return float in [0,1]  
- All modules accept optional series in constructor or as argument to batch methods  
- Rich commentary in docstrings and inline comments explains intended full implementations

---

Online Univariate Structural Break Detector – Full Skeleton with Black-Box Commentary

This skeleton implements all major components from Steps 0 to 5 according to our reference plan and locked-in implementation guidelines. All classes include:
- Role and Step in pipeline  
- Input/Output specifications  
- DL integration points (embeddings)  
- Example behaviour  
- Stub methods to allow iterative implementation

**Notes:**
- All methods are designed for online/sequential updates where applicable  
- Placeholder logic is present where algorithms are to be developed iteratively  
- Commentary is sufficient for an outsider to implement each component independently while preserving interface and dataflow compatibility

---

```python
from __future__ import annotations
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import pandas as pd

# -----------------------------
# Type Aliases
# -----------------------------
Array = np.ndarray

# -----------------------------
# STEP 0: INPUT & PREPROCESSING
# -----------------------------

class Preprocessor:
    """
    Step 0: Classical preprocessing (fully self-explanatory commentary)
    Responsibilities:
      - Prepare raw input series for downstream perspectives by removing deterministic artifacts.
      - Handle three main classical operations:
        1. Detrending
        2. De-seasonalization
        3. Noise reduction / smoothing
    Inputs:
      - series: pd.Series of floats, online or batch.
    Outputs:
      - cleaned pd.Series, same length as input.
    """
    def __init__(self, loess_frac: float = 0.05, seasonal_period: Optional[int] = None):
        self.loess_frac = loess_frac
        self.seasonal_period = seasonal_period

    def denoise(self, series: pd.Series) -> pd.Series:
        """Remove high-frequency noise using LOESS smoothing (placeholder)."""
        return series

    def detrend(self, series: pd.Series) -> pd.Series:
        """Remove slow-moving trend."""
        return series

    def remove_seasonality(self, series: pd.Series) -> pd.Series:
        """Remove periodic / seasonal component if seasonal_period is set."""
        return series

    def preprocess(self, series: pd.Series) -> pd.Series:
        """Full preprocessing pipeline: denoise -> detrend -> remove_seasonality."""
        clean = self.denoise(series)
        clean = self.detrend(clean)
        clean = self.remove_seasonality(clean)
        return clean


class DLPreprocessor:
    """
    Step 0: DL-based preprocessing
    Responsibilities:
      - Provide cleaned point for perspectives
      - Generate embedding vector representing local temporal patterns
    Inputs:
      - x: scalar float
    Outputs:
      - cleaned_point: float
      - embedding: np.ndarray
    """
    def __init__(self, emb_dim: int = 16, window: int = 200):
        self.emb_dim = emb_dim
        self.window = window
        self._buffer: List[float] = []

    def update_buffer(self, x: float):
        self._buffer.append(x)
        if len(self._buffer) > self.window:
            self._buffer.pop(0)

    def transform(self, x: float) -> Tuple[float, Array]:
        self.update_buffer(x)
        cleaned = x - np.mean(self._buffer) if self._buffer else x
        emb = np.random.randn(self.emb_dim).astype(np.float32)
        return cleaned, emb

# -----------------------------
# STEP 1: SHOCK / OUTLIER DETECTION
# -----------------------------

class ShockDetector:
    """
    Step 1: Shock / outlier detection
    Responsibilities:
      - Identify extreme deviations
      - Produce a probabilistic score in [0,1]
    Inputs:
      - x: scalar float
    Outputs:
      - shock_score: float in [0,1]
    """
    def __init__(self, method: str = "zscore", threshold: float = 3.0, window: int = 10):
        self.method = method
        self.threshold = threshold
        self.window = window
        self._buffer: List[float] = []

    def update_buffer(self, x: float):
        self._buffer.append(x)
        if len(self._buffer) > self.window:
            self._buffer.pop(0)

    def update(self, x: float) -> float:
        """Return shock score for new point x."""
        self.update_buffer(x)
        return float(np.random.rand())

# -----------------------------
# STEP 2: PERSPECTIVE MODULES
# -----------------------------

class Perspective:
    """
    Base class for all perspectives.
    Contract:
      - Must implement update(x, embedding, shock_score) -> float
      - Returns score in [0,1]
    """
    def update(self, x: float, embedding: Optional[Array] = None, shock_score: float = 0.0) -> float:
        raise NotImplementedError

class ResidualPerspective(Perspective):
    """
    Step 2: Residual-based perspective

    Responsibilities:
      - Monitor short-term residual dynamics in an online/univariate series.
      - Capture three 'flavours' of potential structural breaks:
        1. Mean shift
        2. Variance shift
        3. Autocorrelation shift
      - Produce a single combined break score in [0,1] per point

    Inputs:
      - x: scalar float
      - embedding: optional np.ndarray
      - shock_score: float in [0,1]

    Outputs:
      - combined: float in [0,1]
    """
    def __init__(self, window: int = 20, lag: int = 1):
        self.window = window
        self.lag = lag
        self._buffer: List[float] = []
        self.last_flavours: Dict[str, float] = {}

    def _mean_shift_score(self) -> float:
        if len(self._buffer) < 2:
            return 0.0
        buf = np.array(self._buffer)
        mean = np.mean(buf[:-1])
        std = np.std(buf[:-1]) + 1e-8
        score = abs(buf[-1] - mean) / std
        return min(score / 3.0, 1.0)

    def _variance_shift_score(self) -> float:
        if len(self._buffer) < 4:
            return 0.0
        buf = np.array(self._buffer)
        var_prev = np.var(buf[:-2])
        var_now = np.var(buf[1:])
        score = abs(var_now - var_prev) / (var_prev + 1e-8)
        return min(score / 3.0, 1.0)

    def _autocorr_shift_score(self) -> float:
        if len(self._buffer) <= self.lag:
            return 0.0
        buf = np.array(self._buffer)
        y1 = buf[self.lag:]
        y2 = buf[:-self.lag]
        ac = np.corrcoef(y1, y2)[0, 1]
        score = abs(1.0 - ac)
        return min(score, 1.0)

    def update(self, x: float, embedding: Optional[Array] = None, shock_score: float = 0.0) -> float:
        self._buffer.append(x)
        if len(self._buffer) > self.window:
            self._buffer.pop(0)

        mean_shift = self._mean_shift_score()
        variance_shift = self._variance_shift_score()
        autocorr_shift = self._autocorr_shift_score()

        combined = (mean_shift + variance_shift + autocorr_shift) / 3.0
        self.last_flavours = {
            "mean_shift": mean_shift,
            "variance_shift": variance_shift,
            "autocorr_shift": autocorr_shift
        }

        return combined


class ThresholdPerspective(Perspective):
    """
    Step 2: Threshold-based perspective

    Responsibilities:
      - Detect points falling outside rolling percentile thresholds.
      - Score is proportional to distance from threshold.

    Inputs:
      - x: scalar float
      - embedding: optional
      - shock_score: float in [0,1]

    Outputs:
      - score: float in [0,1]
    """
    def __init__(self, window: int = 100, percentile: float = 0.95):
        self.window = window
        self.percentile = percentile
        self._buffer: List[float] = []

    def update(self, x: float, embedding: Optional[Array] = None, shock_score: float = 0.0) -> float:
        self._buffer.append(x)
        if len(self._buffer) > self.window:
            self._buffer.pop(0)
        return float(np.random.rand())


class FragilityPerspective(Perspective):
    """
    Step 2: Fragility-based perspective

    Responsibilities:
      - Measure local volatility and autocorrelation growth.

    Inputs:
      - x: scalar float
      - embedding: optional
      - shock_score: float

    Outputs:
      - score: float in [0,1]
    """
    def __init__(self, window: int = 200):
        self.window = window
        self._buffer: List[float] = []

    def update(self, x: float, embedding: Optional[Array] = None, shock_score: float = 0.0) -> float:
        self._buffer.append(x)
        if len(self._buffer) > self.window:
            self._buffer.pop(0)
        return float(np.random.rand())

class LatentStatePerspective(Perspective):
    """
    Step 2: Latent-state / clustering perspective
    Responsibilities:
      - Detect regime shifts by approximating online HMM or clustering.
      - Identify latent states and transitions affecting break likelihood.
    """
    def __init__(self, n_states: int = 2, window: int = 100):
        self.n_states = n_states
        self.window = window
        self._buffer: List[float] = []
        self._centroids: Optional[np.ndarray] = None

    def update(self, x: float, embedding: Optional[Array] = None, shock_score: float = 0.0) -> float:
        self._buffer.append(x)
        if len(self._buffer) > self.window:
            self._buffer.pop(0)
        return float(np.random.rand())


class SurprisePerspective(Perspective):
    """
    Step 2: Surprise / anomaly perspective
    Responsibilities:
      - Detect unusually large deviations in a single point.
      - Can be seen as rolling z-score or percentile outlier detection.
    """
    def __init__(self, window: int = 50):
        self.window = window
        self._buffer: List[float] = []

    def update(self, x: float, embedding: Optional[Array] = None, shock_score: float = 0.0) -> float:
        self._buffer.append(x)
        if len(self._buffer) > self.window:
            self._buffer.pop(0)
        return float(np.random.rand())


class Aggregator:
    """
    Step 3: Aggregates perspective scores
    Responsibilities:
      - Combine scores from all perspectives into a single break probability.
      - Can apply simple max, weighted sum, or Bayesian pooling.
    """
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights

    def aggregate(self, scores: Dict[str, float]) -> float:
        if self.weights:
            wsum = sum(self.weights.values())
            return sum(scores.get(k, 0.0) * self.weights.get(k, 0.0) / wsum for k in scores)
        else:
            return max(scores.values()) if scores else 0.0


class OutputHandler:
    """
    Step 4: Output final break probability
    Responsibilities:
      - Maintain history of per-point scores.
      - Optionally alert if break probability exceeds threshold.
    """
    def __init__(self, alert_threshold: float = 0.8):
        self.alert_threshold = alert_threshold
        self.scores: List[Dict[str, float]] = []
        self.combined_scores: List[float] = []

    def record(self, scores: Dict[str, float], combined: float):
        self.scores.append(scores)
        self.combined_scores.append(combined)
        if combined >= self.alert_threshold:
            print(f"[ALERT] Break probability {combined:.2f} exceeds threshold!")


class PostProcessor:
    """
    Step 5: Optional smoothing / hysteresis
    Responsibilities:
      - Reduce noise and flip-flopping in combined break probability.
      - Apply rolling mean or adaptive thresholds to stabilize output.
    """
    def __init__(self, window: int = 3):
        self.window = window

    def smooth(self, combined_scores: List[float]) -> List[float]:
        smoothed = []
        for i in range(len(combined_scores)):
            start = max(0, i - self.window + 1)
            smoothed.append(np.mean(combined_scores[start:i + 1]))
        return smoothed

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

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # -------------------------------
    # Load example series
    # -------------------------------
    df = pd.read_csv("f:/synth_ts.csv")
    series_ids = df['series_id'].unique()
    random_id = np.random.choice(series_ids)
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
    residual = ResidualPerspective()
    threshold = ThresholdPerspective()
    fragility = FragilityPerspective()
    latent = LatentStatePerspective()
    surprise = SurprisePerspective()
    aggregator = Aggregator()
    output_handler = OutputHandler()
    postprocessor = PostProcessor()

    perspectives = {
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
    # Run pipeline online
    # -------------------------------
    all_scores = {name: [] for name in perspectives}
    combined_scores = []

    for t, x in ts_preprocessed.items():
        x_clean, emb = dlpreprocessor.transform(x)
        shock_score = shock_detector.update(x_clean)

        scores = {}
        for name, perspective in perspectives.items():
            scores[name] = perspective.update(x_clean, embedding=emb, shock_score=shock_score)
            all_scores[name].append(scores[name])

        combined = aggregator.aggregate(scores)
        combined_scores.append(combined)

        smoothed = postprocessor.smooth(combined_scores)
        alert_flag = smoothed[-1] >= output_handler.alert_threshold

        output_handler.record(scores, combined)

    # -------------------------------
    # Plot results
    # -------------------------------
    plt.figure(figsize=(14, 7))
    plt.plot(ts_preprocessed.index, ts_preprocessed.values, label="Preprocessed series", color='black')
    for name, scores_list in all_scores.items():
        plt.plot(ts_preprocessed.index, scores_list, label=f"{name} perspective score")
    plt.plot(ts_preprocessed.index, combined_scores, label="Combined score", linestyle='--', color='red')
    plt.xlabel("Time")
    plt.ylabel("Value / Score")
    plt.legend()
    plt.title(f"Series {random_id} - All perspectives + combined")
    plt.show()
```
