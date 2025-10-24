import numpy as np
from scipy.stats import t, beta

class OnlineHDPHMMChangepointDetector:
    """
    Online Hierarchical Dirichlet Process HMM Changepoint Detector
    -------------------------------------------------------------
    Sequentially estimates the probability that each new observation x_t
    is a change point. Uses a disentangled sticky HDP-HMM with Dirichlet
    Process mixture emissions. Updates parameters online and adapts 
    hyperparameters using Bayesian optimization proxy metrics.
    Stick-breaking construction is used for transitions (Refinement 16).
    """

    def __init__(self, alpha=1.0, gamma=1.0, kappa=0.1, trunc=20, emission_base=None, dim=1):
        # Hyperparameters
        self.alpha = alpha
        self.gamma = gamma
        self.kappa = kappa
        self.trunc = trunc
        self.beta_params = np.full((self.trunc, self.trunc), 0.5)  # stick-breaking Beta parameters
        self.dim = dim
        self.lr_weights = 0.01
        self.lr_alpha = 0.01
        self.lr_kappa = 0.01

        # Mixture weighting --> TODO: update these 3 online
        self.w_new = 0.7 #.5
        self.w_trans = 0.4 #.25
        self.w_emis = 0.4 #.25


        # Run-length posterior
        self.max_run_length = 1000
        self.run_length_probs = np.zeros(self.max_run_length)
        self.run_length_probs[0] = 1.0
        self.hazard = 0.01
        self.df = 5  # t-distribution df

        # Dwell times and stickiness
        self.dwell_time = 0
        self.avg_dwell_time = 1.0
        self.kappa_min, self.kappa_max = 0.01, 5.0
        self.kappa_lr = 0.05

        # Hyperprior parameters
        self.alpha_a, self.alpha_b = 1.0, 1.0
        self.gamma_a, self.gamma_b = 1.0, 1.0
        self.kappa_a, self.kappa_b = 1.0, 1.0

        # Emission prior
        self.emission_base = emission_base or {'mean_prior': np.zeros(dim),
                                              'cov_prior': np.eye(dim) * 5.0}

        # HDP-HMM storage
        self.transition_counts = np.ones((trunc, trunc)) * 1e-3
        self.state_counts = np.zeros(trunc)
        self.state_params = [self._init_state_params() for _ in range(trunc)]
        self.v_sticks = [None] * trunc  # Stick-breaking Beta variables for transitions
        for i in range(trunc):
            self.v_sticks[i] = beta.rvs(1.0, self.gamma, size=self.trunc)

        self.last_state = None
        self.current_states = 1

    def _init_state_params(self):
        K = 5
        state = {
            'weights': np.ones(K) / K,
            'means': [float(np.random.normal(self.emission_base['mean_prior'][0],
                                             np.sqrt(self.emission_base['cov_prior'][0, 0])))
                      for _ in range(K)],
            'covs': [float(self.emission_base['cov_prior'][0, 0]) for _ in range(K)],
            'counts': np.zeros(K, dtype=float)
        }
        return state

    def _posterior_predictive(self, x):
        likelihoods = np.zeros(self.current_states)
        for s_idx, state in enumerate(self.state_params[:self.current_states]):
            mix_likelihood = 0.0
            for k in range(len(state['weights'])):
                w, m, c = state['weights'][k], state['means'][k], state['covs'][k]
                mix_likelihood += w * t.pdf(x, df=self.df, loc=m, scale=np.sqrt(c))
            likelihoods[s_idx] = mix_likelihood
        return likelihoods + 1e-12

    def _update_emission_params(self, state_idx, x):
        state = self.state_params[state_idx]
        K = len(state['weights'])
        pdfs = np.array([t.pdf(x, df=self.df, loc=state['means'][k], scale=np.sqrt(state['covs'][k]))
                         for k in range(K)])
        weighted_pdfs = state['weights'] * pdfs
        r = weighted_pdfs / (weighted_pdfs.sum() + 1e-12)

        for k in range(K):
            state['counts'][k] += r[k]
            n_k = state['counts'][k]
            delta = x - state['means'][k]
            state['means'][k] += r[k] * delta / n_k
            state['covs'][k] += r[k] * (delta**2 - state['covs'][k]) / n_k
        state['weights'] = state['counts'] / (state['counts'].sum() + 1e-12)

    def _update_transition_counts(self, current_state):
        if self.last_state is not None:
            self.transition_counts[self.last_state, current_state] += 1
        self.last_state = current_state

    def _compute_stick_breaking_pi(self, state_idx):
        """
        Compute transition probabilities for a state using stick-breaking construction.
        Ensures array bounds are respected and extends beta_params if needed.
        """
        # Extend beta_params if current_states exceeds trunc
        if self.current_states > self.trunc:
            extend = self.current_states - self.trunc
            self.beta_params = np.pad(
                self.beta_params,
                ((0, extend), (0, extend)),
                'constant',
                constant_values=0.5
            )
            self.trunc += extend

        # Use only the relevant part of beta_params
        v = self.beta_params[state_idx, :self.current_states]
        pi = np.zeros(self.current_states)
        prod = 1.0
        for k in range(self.current_states):
            pi[k] = v[k] * prod
            prod *= (1 - v[k])
        # Ensure normalization due to numerical issues
        pi /= pi.sum() + 1e-12
        return pi


    def _allocate_new_state(self, x, p_new=None):
        """
        Allocate a new HDP-HMM state and extend truncation arrays if necessary.
        """
        # Extend truncation if needed
        if self.current_states >= self.trunc:
            extend_size = max(5, self.current_states - self.trunc + 1)
            self.trunc += extend_size
            self.transition_counts = np.pad(
                self.transition_counts, ((0, extend_size), (0, extend_size)), 'constant', constant_values=1e-3
            )
            self.state_counts = np.pad(self.state_counts, (0, extend_size), 'constant')
            self.state_params.extend([self._init_state_params() for _ in range(extend_size)])
            if hasattr(self, 'beta_params'):
                self.beta_params = np.pad(
                    self.beta_params, ((0, extend_size), (0, extend_size)), 'constant', constant_values=0.5
                )
            else:
                self.beta_params = np.full((self.trunc, self.trunc), 0.5)

        # Allocate the new state
        new_state_idx = self.current_states
        self.current_states += 1
        self.state_params[new_state_idx] = self._init_state_params()

        # Initialize stick-breaking beta_params row/column for new state
        self.beta_params[new_state_idx, :self.current_states] = 0.5
        self.beta_params[:self.current_states, new_state_idx] = 0.5

        return new_state_idx


    def _update_run_length(self, x, state_likelihoods):
        pred_likelihood = state_likelihoods.sum() + 1e-12
        growth_probs = (1 - self.hazard) * self.run_length_probs[:self.current_states] * pred_likelihood
        cp_prob = self.hazard * self.run_length_probs[:self.current_states] * pred_likelihood
        new_run_length_probs = np.zeros_like(self.run_length_probs)
        new_run_length_probs[1:self.current_states + 1] = growth_probs
        new_run_length_probs[0] = cp_prob.sum()
        new_run_length_probs /= new_run_length_probs.sum() + 1e-12
        self.run_length_probs = new_run_length_probs
        return self.run_length_probs[0]

    def _update_hyperparams_vi(self, grad_alpha, grad_gamma, grad_kappa, lr=0.01):
        self.alpha = max(1e-3, self.alpha + lr * grad_alpha)
        self.gamma = max(1e-3, self.gamma + lr * grad_gamma)
        self.kappa = max(1e-3, self.kappa + lr * grad_kappa)

    def _resample_hyperparams(self):
        eta = np.random.beta(self.alpha + 1, self.state_counts.sum())
        pi_eta = (self.alpha_a + self.current_states - 1) / (
            self.state_counts.sum() * (self.alpha_b - np.log(eta)) + self.alpha_a + self.current_states - 1
        )
        if np.random.rand() < pi_eta:
            self.alpha = np.random.gamma(self.alpha_a + self.current_states, 1.0 / (self.alpha_b - np.log(eta)))
        else:
            self.alpha = np.random.gamma(self.alpha_a + self.current_states - 1, 1.0 / (self.alpha_b - np.log(eta)))

    def update(self, x):
        """
        Feed a new observation x_t and get refined change point probability.
        Implements:
          - Posterior predictive over DP mixture emissions
          - Stick-breaking construction for transition distributions (CRF)
          - Sticky-HDP stickiness adjustment
          - Run-length posterior contribution
          - Online emission and hyperparameter updates
        """
        # --- Step 1: Posterior predictive for existing states ---
        state_likelihoods = self._posterior_predictive(x)
        p_cp_run_length = self._update_run_length(x, state_likelihoods)

        # --- Step 2: New state probability (CRF-inspired) ---
        total_count = self.state_counts[:self.current_states].sum()
        p_new = self.alpha * self.gamma / (self.gamma + total_count)

        # --- Step 3: Normalize ---
        total = state_likelihoods.sum() + p_new
        state_probs = state_likelihoods / total
        p_new_norm = p_new / total

        # --- Step 4: Assign state ---
        all_probs = np.append(state_probs, p_new)
        assigned_state = np.argmax(all_probs)
        is_new_state = assigned_state == self.current_states

        # --- Step 5: Allocate new state if needed ---
        if is_new_state:
            assigned_state = self._allocate_new_state(x)

        # --- Step 6: Update dwell times ---
        if self.last_state is None or assigned_state != self.last_state:
            self.avg_dwell_time = 0.9 * self.avg_dwell_time + 0.1 * self.dwell_time
            self.dwell_time = 1
        else:
            self.dwell_time += 1

        # --- Step 7: Compute stick-breaking transition distribution ---
        # After new state allocation, current_states updated
        pis = np.array([self._compute_stick_breaking_pi(i) for i in range(self.current_states)])
        
        if self.last_state is None:
            trans_shift = 1.0
        else:
            last = self.last_state
            sticky_prior = pis[last, :self.current_states] + self.kappa * (np.arange(self.current_states) == last)
            numerator = sticky_prior[assigned_state]
            denominator = sticky_prior.sum()
            trans_prob = numerator / denominator
            trans_shift = 1.0 - np.clip(trans_prob, 0.0, 1.0)

        # --- Step 8: Emission likelihood shift ---
        if self.last_state is None:
            emis_shift = 1.0
        else:
            prev_state = self.state_params[self.last_state]
            prev_likelihood = sum(
                w * t.pdf(x, df=self.df, loc=m, scale=np.sqrt(c))
                for w, m, c in zip(prev_state['weights'], prev_state['means'], prev_state['covs'])
            ) + 1e-12

            assigned_params = self.state_params[assigned_state]
            assigned_likelihood = sum(
                w * t.pdf(x, df=self.df, loc=m, scale=np.sqrt(c))
                for w, m, c in zip(assigned_params['weights'], assigned_params['means'], assigned_params['covs'])
            ) + 1e-12

            emis_shift = 1.0 - prev_likelihood / (prev_likelihood + assigned_likelihood)
            #emis_shift = np.log(assigned_likelihood + 1e-12) - np.log(prev_likelihood + 1e-12)
            #along with the preceding line: emis_shift = 1 / (1 + np.exp(-emis_shift))  # sigmoid to 0-1



        # --- Step 9: Combine factors into p_cp with online weights ---
        p_cp = self.w_new * p_new_norm + self.w_trans * trans_shift + self.w_emis * emis_shift
        error_signal = p_new_norm - p_cp

        self.w_new  += self.lr_weights * error_signal * p_new_norm
        self.w_trans += self.lr_weights * error_signal * trans_shift
        self.w_emis += self.lr_weights * error_signal * emis_shift
        total_w = self.w_new + self.w_trans + self.w_emis + 1e-12
        self.w_new  /= total_w
        self.w_trans /= total_w
        self.w_emis  /= total_w

        # Include run-length posterior contribution
        lambda_rl = 0.3
        p_cp = (1 - lambda_rl) * p_cp + lambda_rl * p_cp_run_length
        p_cp = np.clip(p_cp, 0.0, 1.0)

        # --- Step 10: Update transitions and emissions ---
        self._update_transition_counts(assigned_state)
        self.state_counts[assigned_state] += 1
        self._update_emission_params(assigned_state, x)

        # --- Step 11: Online hyperparameter updates ---
        grad_alpha = - (p_new_norm - 0.1)
        grad_gamma = emis_shift - 0.5
        grad_kappa = trans_shift - 0.5
        self._update_hyperparams_vi(grad_alpha, grad_gamma, grad_kappa, lr=0.01)

        if p_new > 0.3:
            self._resample_hyperparams()

        # --- Step 12: Adapt kappa for dwell time ---
        if self.dwell_time > self.avg_dwell_time:
            self.kappa = min(self.kappa * (1 + self.kappa_lr), self.kappa_max)
        else:
            self.kappa = max(self.kappa * (1 - self.kappa_lr), self.kappa_min)

        return p_cp

