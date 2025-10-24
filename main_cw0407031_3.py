from support_cw0407011_13 import OnlinePreprocessor, OldPreprocessor, ShockDetector, SurprisePerspective, PostProcessor, OutputHandler
#from cdt_cw0407021_1 import CUSUMDetection, EWMAVarianceDetection, Aggregator, BinSegDetection, StickyOnlineHDPHMMDetection, BOCPDDetection
from dshdphmm_45 import OnlineHDPHMMChangepointDetector
#best performance: 21, last good version: 39
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.signal import savgol_filter
from statsmodels.tsa.seasonal import STL
from typing import List, Dict, Optional, Any, Tuple
from collections import deque
import random


"""
data:
    non-stationary
    non-i.i.d --> preprocessing?
    univariate/n-dimensional
    continuous

change point detection:
    non-parametric
    online

"""
"""
">": adopt

==preprocessing
	denoise, detrend, periodic behaviour?, deseasonalise (..Change Point Detection for Random Objects with Possibly Periodic Behavior)
	kalman filter (..as preprocessor --> yes, for detrending/smoothing)

==shock detection

==anomaly detection

==detection layer (Hierarchical Change-Detection Tests)
	epsilon machines suitable for discrete event streams (not here) (v kalman filter ..as concept drift change detectors)
                                    (...neither: kalman filter is very parametric --except in Bayesian Markov)
		(Deciphering hierarchical features in the energy landscape of adenylate kinase folding/unfolding)
	>bayesian or infinite hidden markov models***
                                    (non-gaussian, Bayesian Hidden Markov Models:                                    
                                    Bayesian autoregressive models.
                                    Dirichlet Process Mixture State-Space Models: Infinite HMMs (iHMMs), Sticky HDP-HMMs,
                                            > [sticky] Hierarchical Dirichlet Process Hidden Markov Models (HDP-HMM) or HDP-AR --> these are mostly offline or semi-online
	>data-driven (instead of model-based): statistics or unsupervised learning
                                    >BOCPD (Adams, MacKay, 2007)
                                    >At Most One Change (Hinkley, 1970)
                                    CPNP (Haynes et al., 2017)
                                    Binary segmentation (Scott and Knott, 1974) --> not really online
                                    >Energy change point (Matteson and James, 2014)
                                    pelt: Pruned Exact Linear Time (Killick et al., 2012)
                                    Robust Functional Pruning Optimal Partitioning (Fearnhead and Rigaill, 2019)
                                    Segment Neighborhoods (Auger and Lawrence, 1989)
                                    

	>? anytime or contract algorithm for stopping the algorithm: "anytime algorithms [28] which allow the execution to be interrupted at any time
                                                                                                    and output the best possible solution obtained so far. A similar method is a contract algorithm which
                                                                                                    also trades off computation time for solution quality but is given the allowable run time in advance
                                                                                                    as a type of contract agreement."

==validation layer (Hierarchical Change-Detection Tests)
	...


==aggregation

==post processing
        smoothing?
	(Predictive State Smoothing (PRESS): Scalable non-parametric regression for high-dimensional data with variable selection)

==output handling



+ (HHS Public Access A Survey of Methods for Time Series Change Point Detection)
++ (An Evaluation of Change Point Detection Algorithms)


***
Bayesian Markov Models
│
├── 1. Discrete-State Models
│   │
│   ├── Finite HMMs (Bayesian Hidden Markov Models)
│   │   • Latent states are discrete
│   │   • Transition matrix has a prior (Dirichlet)
│   │   • Emissions can be Gaussian, non-Gaussian
│   │
│   └── Nonparametric HMMs
│       ├── Infinite HMM (iHMM)
│       │   • Hierarchical Dirichlet Process (HDP) prior
│       │   • Unbounded number of hidden states
│       └── Sticky HDP-HMM
│           • Adds persistence to regimes
│           • Common in finance, neuroscience
│
├── 2. Continuous-State Models
│   │
│   ├── Linear-Gaussian (special case)
│   │   └── Kalman Filter
│   │       • Latent state evolves linearly with Gaussian noise
│   │       • Closed-form Bayesian filtering
│   │
│   ├── Nonlinear/Non-Gaussian
│   │   ├── Extended/Unscented Kalman Filters
│   │   └── Particle Filters (Sequential Monte Carlo)
│   │       • Posterior approximated by particles
│   │       • Fully nonparametric, nonlinear, non-Gaussian
│   │
│   └── Bayesian Nonparametric Continuous Models
│       ├── Gaussian Process State-Space Models (GP-SSM)
│       │   • Transition/observation dynamics drawn from GP priors
│       │   • Very flexible, nonparametric
│       └── Bayesian AR / GARCH with nonparametric priors
│
└── 3. Change-Point Focused Bayesian Models
    │
    ├── Bayesian Online Change Point Detection (BOCPD)
    │   • Run-length as latent Markov process
    │   • Detects structural breaks online
    │
    └── Regime-Switching Models
        • Hidden regime is a Markov chain
        • Break = switch in hidden regime
                    e.g:
                    >Smooth Transition Autoregressive (STAR) models: transition between regimes is smooth (via logistic/sigmoid), not abrupt.
                    Threshold Autoregressive (TAR) models: regime switches occur when an observable variable (e.g., lagged value of the series) crosses a threshold.
                    >Markov-Switching Autoregressive (MS-AR) models: AR coefficients differ by regime.
                    >Markov-Switching GARCH (MS-GARCH): volatility dynamics switch across regimes.
                    >Markov-Switching VAR: multivariate extension for economic/financial systems.
                    ...

"""

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
    df = pd.read_csv("f:/synth_ts_3.csv")
    df['is_break'] = df['is_break'].astype(bool)
    df['is_shock'] = df['is_shock'].astype(bool)

    series_ids = df['series_id'].unique()
    random_id = np.random.choice(series_ids)
    #random_id = "ecology_set1"
    random_id = "climatology_set1"
    random_series = df[df['series_id'] == random_id].sort_values("time")
    break_times = random_series.loc[random_series['is_break'], 'time'].values
    shock_df   = random_series.loc[random_series['is_shock'], ['time', 'value']]
    ts = pd.Series(data=random_series['value'].values, index=random_series['time'])
    print(f"Randomly selected series: {random_id}")
    print(ts.head())

    # -------------------------------
    # Initialize modules
    # -------------------------------
    preprocessor = OnlinePreprocessor(
        detrend=True,
        denoise_alpha=0.2,  # set to None to disable
        season_window=50     # rolling window for adaptive seasonal component
    )
    oldpreprocessor=OldPreprocessor()
    #dlpreprocessor = DLPreprocessor()
    shock_detector = ShockDetector()
    #bocpd = BOCPDDetection()
    #residual = ResidualPerspective()
    #cusum=CUSUMDetection()
    #ewma=EWMAVarianceDetection()
    #binseg=BinSegDetection()
    dshdphmm=OnlineHDPHMMChangepointDetector()
    #threshold = ThresholdPerspective()
    #fragility = FragilityPerspective()
    #latent = LatentStatePerspective()
    surprise = SurprisePerspective()
    #aggregator = Aggregator()
    output_handler = OutputHandler()
    postprocessor = PostProcessor()

    # Dictionary of perspectives for easy iteration
    perspectives = {
        #'bocpd': bocpd,
        #'cusum': cusum,
        #'ewma': ewma,
        #'binseg':binseg,
        'dshdphmm':dshdphmm,
        #'threshold': threshold,
        #'fragility': fragility,
        #'latent': latent,
        #'surprise': surprise
    }

    # -------------------------------
    # Preprocess series
    # -------------------------------
    #ts_preprocessed = preprocessor.preprocess(ts)
    ts_old_preproc = oldpreprocessor.preprocess(ts) # only for comparison with the new version

    # -------------------------------
    # Prepare storage for scores
    # -------------------------------
    all_scores = {name: [] for name in perspectives}
    #combined_scores = []
    shock_scores = []
    residuals = []

    # -------------------------------
    # Run pipeline online
    # -------------------------------
    for t, x in ts.items(): #ts_preprocessed.items():
        # online preprocessing
        x_clean = preprocessor.update(x)
        residuals.append(x_clean)


        # DL preprocessing: cleaned point + embedding
        #x_clean, emb = dlpreprocessor.transform(x)

        # Shock detection: downweight extreme points
        shock_score = shock_detector.update(x_clean)
        shock_scores.append(shock_score) 

        # Update perspectives
        scores = {}
        for name, perspective in perspectives.items():
            scores[name] = perspective.update(x_clean) #, embedding=emb, shock_score=shock_score)
            all_scores[name].append(scores[name])

        # Aggregate perspective scores into combined break probability
        #aggregator.calibrate_from_synthetic(emb_dim=len(emb))
        #combined = aggregator.aggregate(scores, embedding=emb)
        #combined_scores.append(combined)

        # Post-process combined score (smoothing)
        #smoothed = postprocessor.smooth(combined_scores)
        #alert_flag = smoothed[-1] >= output_handler.alert_threshold

        # Record outputs
        #output_handler.record(scores, combined)


    # -------------------------------
    # Plot results
    # -------------------------------
    plt.figure(figsize=(14,7))
    plt.plot(ts, label="Original", color='#000055')
    plt.plot(ts_old_preproc.index, ts_old_preproc.values, label="Old version: Preprocessed series", color='#005500')
    plt.plot(residuals, label="Preprocessed Residuals", color='#550000')
    for name, scores_list in all_scores.items():
        plt.plot(ts.index, scores_list, label=f"{name} perspective score", color='#ff0000') #color="#{:06x}".format(random.randint(0, 0xFFFFFF)))
    #plt.plot(ts_preprocessed.index, combined_scores, label="Combined score", linestyle='--', color='red')
    plt.plot(ts.index, shock_scores, label="Shock detector score", linestyle='-.', color='orange')
    plt.axhline(y=1.0, color='blue', linestyle=':', label='y = 1.0')
    # mark true breaks
    for i, bt in enumerate(break_times):
        plt.axvline(bt, color='gray', linestyle='--', label='True Break' if i == 0 else "")
    # mark true shocks
    for i, st in enumerate(shock_df['time']):
        plt.axvline(
            x=st,
            color='#DDBBBB',
            linestyle=':',
            label='True Shock' if i == 0 else ""
        )

    plt.xlabel("Time")
    plt.ylabel("Value / Score")
    plt.legend()
    #plt.title(f"Series {random_id} - All perspectives + combined")
    plt.show()
