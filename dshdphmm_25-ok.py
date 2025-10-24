import numpy as np
from scipy.stats import norm

class OnlineHDPHMMChangepointDetector:
    """
    Online Hierarchical Dirichlet Process HMM Changepoint Detector
    -------------------------------------------------------------
    Sequentially estimates the probability that each new observation x_t
    is a change point. Uses a Disentangled Sticky HDP-HMM with Dirichlet
    Process mixture emissions. Updates parameters online and adapts 
    hyperparameters using Bayesian optimization proxy metrics.
    """

    def __init__(self, alpha=1.0, gamma=1.0, kappa=0.1, trunc=20, emission_base=None, dim=1):
        self.alpha = alpha
        self.gamma = gamma
        self.kappa = kappa
        self.trunc = trunc
        self.dim = dim
        self.lr_alpha = 0.01
        self.lr_kappa = 0.01
        self.w_new = 0.5
        self.w_trans = 0.25
        self.w_emis = 0.25
        self.lr_weights = 0.01 # Learning rate for weights adaptation
        self.max_run_length = 1000  # maximum tracked run-length
        self.run_length_probs = np.zeros(self.max_run_length)
        self.run_length_probs[0] = 1.0  # at t=0, run-length = 0, represents P(r0=r)
        self.hazard = 0.01  # constant hazard rate (can be adapted)


        
        self.emission_base = emission_base or {'mean_prior': np.zeros(dim),
                                              'cov_prior': np.eye(dim) * 5.0}
        
        # Transition counts and state counts
        self.transition_counts = np.ones((trunc, trunc)) * 1e-3
        self.state_counts = np.zeros(trunc)
        self.state_params = [self._init_state_params() for _ in range(trunc)]
        
        self.last_state = None
        self.current_states = 1  # track actual number of allocated states

    def _init_state_params(self):
        K = 5  # mixture components per state
        state = {
            'weights': np.ones(K) / K,
            'means': [float(np.random.normal(self.emission_base['mean_prior'][0],
                                             np.sqrt(self.emission_base['cov_prior'][0,0])))
                      for _ in range(K)],
            'covs': [float(self.emission_base['cov_prior'][0,0]) for _ in range(K)],
            'counts': np.zeros(K, dtype=float)
        }
        return state

    def _posterior_predictive(self, x):
        likelihoods = np.zeros(self.current_states)
        for s_idx, state in enumerate(self.state_params[:self.current_states]):
            mix_likelihood = 0.0
            for k in range(len(state['weights'])):
                weight = state['weights'][k]
                mean = state['means'][k]
                cov = state['covs'][k]
                mix_likelihood += weight * norm.pdf(x, loc=mean, scale=np.sqrt(cov))
            likelihoods[s_idx] = mix_likelihood
        return likelihoods + 1e-12

    def _new_state_probability(self, x):
        mean = self.emission_base['mean_prior'][0]
        cov = self.emission_base['cov_prior'][0,0]
        # Small prior to favor new state only when necessary
        return max(norm.pdf(x, loc=mean, scale=np.sqrt(cov)), 1e-3)

    def _update_emission_params(self, state_idx, x):
        """
        Online DP mixture emission update using responsibilities (soft assignment).
        Implements truncated variational inference updates for Gaussian mixture.
        Reference: Fox et al., 2011; Hoffman et al., 2013 (online VI)
        
        Parameters
        ----------
        state_idx : int
            Index of the HDP-HMM state to update
        x : float
            New observation (scalar)
        """
        state = self.state_params[state_idx]
        K = len(state['weights'])
        
        # Step 1: Compute responsibilities (soft assignments)
        pdfs = np.array([norm.pdf(x, loc=state['means'][k], scale=np.sqrt(state['covs'][k]))
                         for k in range(K)])
        weighted_pdfs = state['weights'] * pdfs
        r = weighted_pdfs / (weighted_pdfs.sum() + 1e-12)
        
        # Step 2: Update sufficient statistics
        for k in range(K):
            # Soft count update
            state['counts'][k] += r[k]
            n_k = state['counts'][k]
            
            # Mean update (scalar)
            delta = x - state['means'][k]
            state['means'][k] += r[k] * delta / n_k
            
            # Variance update (scalar)
            state['covs'][k] += r[k] * (delta**2 - state['covs'][k]) / n_k
        
        # Step 3: Update mixture weights
        state['weights'] = state['counts'] / (state['counts'].sum() + 1e-12)


    def _update_transition_counts(self, current_state):
        if self.last_state is not None:
            self.transition_counts[self.last_state, current_state] += 1 + self.kappa
        self.last_state = current_state

    def _allocate_new_state(self, x):
        """
        Dynamically allocate a new HDP-HMM state.
        - Extends truncation if needed
        - Initializes emission DP mixture parameters
        - Updates transition counts placeholder
        
        Returns
        -------
        new_state_idx : int
        """
        """
        Allocate a new HDP-HMM state. Only allocates; assignment is handled in update().
        """
        if self.current_states >= self.trunc:
            extend_size = 5
            self.trunc += extend_size
            self.transition_counts = np.pad(self.transition_counts, ((0, extend_size),(0,extend_size)), 'constant')
            self.state_counts = np.pad(self.state_counts, (0,extend_size), 'constant')
            self.state_params.extend([self._init_state_params() for _ in range(extend_size)])

        new_state_idx = self.current_states
        self.current_states += 1
        self.state_params[new_state_idx] = self._init_state_params()
        return new_state_idx



    def _adapt_hyperparameters(self, assigned_state, p_new_norm):
        """
        Stochastic gradient ascent update of hyperparameters alpha and kappa.
        assigned_state : int
            The state assigned to current observation
        p_new_norm : float
            Posterior predictive probability of a new state
        """
        # Gradient wrt alpha: encourage higher alpha if new state probability is high
        grad_alpha = p_new_norm - 0.5   # target 0.5
        self.alpha = max(1e-3, self.alpha + self.lr_alpha * grad_alpha)

        # Gradient wrt kappa: encourage higher kappa if self-transition probability is high
        last = self.last_state
        if last is not None:
            trans_prob = (self.transition_counts[last, assigned_state] + self.kappa * (last==assigned_state)) / \
                         (self.transition_counts[last, :self.current_states].sum() + self.kappa)
            grad_kappa = (last == assigned_state) * (trans_prob - 0.5)
            self.kappa = np.clip(self.kappa + self.lr_kappa * grad_kappa, 1e-3, 1.0)


    def _update_run_length(self, x, state_likelihoods):
        """
        Propagate run-length posterior.
        """
        # 1. Predictive likelihood of x under current run lengths
        pred_likelihood = state_likelihoods.sum() + 1e-12  # sum over states

        # 2. Compute growth probabilities (run continues)
        growth_probs = (1 - self.hazard) * self.run_length_probs[:self.current_states] * pred_likelihood

        # 3. Compute changepoint probability (run resets)
        cp_prob = self.hazard * self.run_length_probs[:self.current_states] * pred_likelihood

        # 4. Shift run lengths
        new_run_length_probs = np.zeros_like(self.run_length_probs)
        new_run_length_probs[1:self.current_states+1] = growth_probs
        new_run_length_probs[0] = cp_prob.sum()  # reset for new run

        # Normalize
        new_run_length_probs /= new_run_length_probs.sum() + 1e-12

        self.run_length_probs = new_run_length_probs

        # Posterior probability of a change point
        p_cp_run_length = self.run_length_probs[0]

        return p_cp_run_length


    def update(self, x):
        """
        Feed a new observation x_t and get refined change point probability.
        Combines:
          - Posterior predictive of new state
          - Transition shift from previous state
          - Emission likelihood shift from previous state
        """
        # Step 1: Posterior predictive for existing states
        state_likelihoods = self._posterior_predictive(x)
        p_cp_run_length = self._update_run_length(x, state_likelihoods)

        # Step 2: New state probability using CRF
        total_count = self.state_counts[:self.current_states].sum()
        p_new = self.alpha * self.gamma / (self.gamma + total_count)  # CRF-inspired

        # Step 3: Normalize with existing states
        total = state_likelihoods.sum() + p_new
        state_probs = state_likelihoods / total
        p_new_norm = p_new / total

        # Step 4: Assign state (argmax)
        all_probs = np.append(state_probs, p_new)
        assigned_state = np.argmax(all_probs)
        is_new_state = assigned_state == self.current_states
        if is_new_state:
            assigned_state = self._allocate_new_state(x)  # just allocates and returns new_state_idx
        
        # Step 5: Transition shift
        if self.last_state is None:
            trans_shift = 1.0
        else:
            last = self.last_state
            trans_prob = (self.transition_counts[last, assigned_state] + self.kappa * (last==assigned_state)) / \
                         (self.transition_counts[last, :self.current_states].sum() + self.kappa)
            trans_shift = 1.0 - np.clip(trans_prob, 0.0, 1.0)
        
        # Step 6: Emission likelihood shift
        if self.last_state is None:
            emis_shift = 1.0
        else:
            prev_state = self.state_params[self.last_state]
            prev_likelihood = sum(
                w * norm.pdf(x, loc=m, scale=np.sqrt(c))
                for w, m, c in zip(prev_state['weights'], prev_state['means'], prev_state['covs'])
            ) + 1e-12
            # Likelihood under assigned state
            assigned_params = self.state_params[assigned_state]
            assigned_likelihood = sum(
                w * norm.pdf(x, loc=m, scale=np.sqrt(c))
                for w, m, c in zip(assigned_params['weights'], assigned_params['means'], assigned_params['covs'])
            ) + 1e-12
            # Shift factor: low likelihood under previous state relative to current → high shift
            emis_shift = 1.0 - prev_likelihood / (prev_likelihood + assigned_likelihood)
        
        # Step 7: Combine factors into p_cp
        p_cp = 0.5 * p_new_norm + 0.25 * trans_shift + 0.25 * emis_shift # a crude, arbitrary weighting to initialise
        # Compute proxy error signal (can use p_new_norm)
        error_signal = p_new_norm - p_cp

        # Gradient-like weight update
        self.w_new  += self.lr_weights * error_signal * p_new_norm
        self.w_trans += self.lr_weights * error_signal * trans_shift
        self.w_emis += self.lr_weights * error_signal * emis_shift

        # Normalize weights to sum to 1 and clip to [0,1]
        total_w = self.w_new + self.w_trans + self.w_emis + 1e-12
        self.w_new  /= total_w
        self.w_trans /= total_w
        self.w_emis  /= total_w

        p_cp = self.w_new * p_new_norm + self.w_trans * trans_shift + self.w_emis * emis_shift
        # Add run-length posterior contribution
        lambda_rl = 0.3  # weight for run-length posterior
        p_cp = (1 - lambda_rl) * p_cp + lambda_rl * p_cp_run_length
        p_cp = np.clip(p_cp, 0.0, 1.0)
        
        # Step 8: Update transition counts and emissions
        self._update_transition_counts(assigned_state)
        self.state_counts[assigned_state] += 1
        self._update_emission_params(assigned_state, x)
        
        # Step 9: Optional hyperparameter adaptation
        self._adapt_hyperparameters(assigned_state, p_new_norm)
        
        return p_cp
