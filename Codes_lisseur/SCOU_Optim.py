class SCOU():

    def __init__(self, observations, lod_vect,                                                          # SCOU's inputs
                 nb_states,                                                                             # SCOU's hyperparameter
                 solver='Nelder-Mead', opt_tol=5e-3, max_iter=600, adaptive=True, disp=True):           # Nelder Mead's parameters

        self.observations = observations
        self.lod_vect = lod_vect
        self.nb_states = nb_states
        self.solver = solver
        self.opt_tol = opt_tol
        self.max_iter = max_iter
        self.adaptive = adaptive
        self.disp = disp

        self.init_parameters_vertex()
        self.get_T_ronde()
        self.get_X_ronde()
        self.observations_below_LoD = np.where(self.observations<=self.lod_vect)
        self.observations_above_LoD = np.setdiff1d(self.T_ronde, self.observations_below_LoD)
        self.get_outlier_emission_parts()

    def init_parameters_vertex(self):
        self.epsilon = np.nanstd(self.observations)
        self.sigma = self.epsilon * 0.5
        self.p_out = 0.1
    
    def get_T_ronde(self):
        self.T_ronde = np.where(~np.isnan(self.observations))[0]
        
    def get_X_ronde(self):
        self.obs_std = np.nanstd(self.observations)
        self.borne_inf, self.borne_sup = np.nanmin(self.observations) - 2*self.obs_std, np.nanmax(self.observations) + 2*self.obs_std

        self.discretization_step = (self.borne_sup - self.borne_inf) / self.nb_states
        self.X_ronde = np.arange(self.borne_inf, self.borne_sup, self.discretization_step)
        self.nb_states = self.X_ronde.shape[0] # because of how np.arange works, this can sometimes lead to an extra state, which is acceptable
    
    def compute_transition_matrix(self):
        self.transition_matrix = ss.norm.pdf(self.X_ronde[:, np.newaxis], self.X_ronde, self.sigma)
        self.transition_matrix /= self.transition_matrix.sum(axis=1)[:, np.newaxis]

        self.transition_matrix = np.log(self.transition_matrix)
        self.matMax = self.transition_matrix.max()
        self.transition_matrix = np.exp(self.transition_matrix - self.matMax)

    def get_outlier_emission_parts(self):
        self.outlier_part_censored = (self.lod_vect[self.observations_below_LoD] - self.borne_inf) / (self.borne_sup - self.borne_inf)
        self.outlier_part_uncensored = np.ones(self.observations_above_LoD.shape[0]) / (self.borne_sup - self.borne_inf)
    
    def compute_emission_matrix(self):
        self.emission_matrix = np.empty((self.observations.shape[0], self.X_ronde.shape[0]))
        self.emission_matrix[:] = np.nan

        self.emission_matrix[self.observations_above_LoD] = (1 - self.p_out) * ss.norm.pdf(self.observations[self.observations_above_LoD, np.newaxis], self.X_ronde, self.epsilon)
        self.emission_matrix[self.observations_above_LoD] += self.p_out * self.outlier_part_uncensored[:, np.newaxis]

        self.emission_matrix[self.observations_below_LoD] = (1 - self.p_out) * ss.norm.cdf(self.lod_vect[self.observations_below_LoD, np.newaxis], self.X_ronde, self.epsilon)
        self.emission_matrix[self.observations_below_LoD] += self.p_out * self.outlier_part_censored[:, np.newaxis]

        self.emission_matrix[np.where(np.isnan(self.observations))] = 1.0 # dealing with unobserved timestamps

        self.emission_matrix = np.log(self.emission_matrix)
        self.rowMax = self.emission_matrix.max(axis=1).reshape(-1,1)
        self.rowMax[np.where(np.isnan(self.observations))] = 0
        self.emission_matrix -= self.rowMax
        self.emission_matrix = np.exp(self.emission_matrix)
        
    def compute_forward_matrix(self):
        self.compute_emission_matrix()
        self.compute_transition_matrix()

        self.forward_matrix = np.empty((self.observations.shape[0], self.X_ronde.shape[0]))
        self.forward_matrix[:] = np.nan

        tmp = self.emission_matrix[0].max()
        self.forward_matrix[0] = self.emission_matrix[0] / tmp
        
        self.L = np.empty(self.observations.shape[0])
        self.L.fill(np.nan)
        self.L[0] = self.rowMax[0] + np.log(tmp) 

        for t in range(1, self.observations.shape[0]):
            A = self.forward_matrix[t-1]
            B = self.transition_matrix
            C = self.emission_matrix[t]
            D = np.dot(A, B)

            self.forward_matrix[t] = D * C

            tmp = self.forward_matrix[t].max()
            self.forward_matrix[t] /= tmp
            self.L[t] = self.L[t-1] + self.rowMax[t] + self.matMax + np.log(tmp)

    def compute_backward_matrix(self):
        self.compute_emission_matrix()
        self.compute_transition_matrix()

        self.backward_matrix = np.empty((self.observations.shape[0], self.X_ronde.shape[0]))
        self.backward_matrix[:] = np.nan     

        self.backward_matrix[-1] = np.ones(self.backward_matrix.shape[1])
        for t in reversed(range(0, self.observations.shape[0]-1)):
            A = self.backward_matrix[t+1]
            B = self.transition_matrix
            C = self.emission_matrix[t+1]

            self.backward_matrix[t] = np.dot(B, A*C)
            tmp = self.backward_matrix[t].max()
            self.backward_matrix[t] /= tmp

    def compute_log_likelihood(self, params):
        self.sigma, self.epsilon, self.p_out = params
        self.compute_forward_matrix()
        LL = self.L[-1] + np.log(self.forward_matrix[-1].sum())

        return -LL
        
    def fit(self):
        initial_guess = [self.sigma, self.epsilon, self.p_out]
        result = minimize(fun=self.compute_log_likelihood, x0=initial_guess, method=self.solver, 
                          options={'maxiter':self.max_iter, 'xatol':self.opt_tol, 'disp':self.disp, 'adaptive': self.adaptive})
        
    def predict(self):
        self.compute_forward_matrix()
        self.compute_backward_matrix()

        self.post_proba_matrix = self.forward_matrix * self.backward_matrix
        self.post_proba_matrix /= self.post_proba_matrix.sum(axis=1)[:, np.newaxis]
        self.muX = np.dot(self.post_proba_matrix, self.X_ronde)

        self.cum_post_proba_matrix = np.cumsum(self.post_proba_matrix, axis=1)
        self.IC95_lower = self.X_ronde[[np.argmin(np.abs(cum_post_proba-.025)) for cum_post_proba in self.cum_post_proba_matrix]]
        self.IC95_upper = self.X_ronde[[np.argmin(np.abs(cum_post_proba-.975)) for cum_post_proba in self.cum_post_proba_matrix]]

    def compute_trajectories(self, Nsim=100):
        # Output instantiation 
        trajectories_matrix = (np.ones((self.observations.shape[0], Nsim))*np.nan)
        self.sigX = (self.IC95_upper - self.IC95_lower)/4
        # For each simulation :
        # 1 - Initialize the trajectory by drawing the first point in a normal distribution of parameters self.muX[0] and self.sigX[0]
        # 2 - Find the corresponding discretized state of that point
        # 3 - For the remaining points, find the most likely state to transition to from the previous state based on the presence of an observation at the time step of interest
        for nsim in range(Nsim):
            trajectory = np.zeros(self.observations.shape[0])
            trajectory[0] = np.random.normal(self.muX[0], self.sigX[0], 1)[0]
            discretized_state = np.argmin(np.abs(trajectory[0] - self.X_ronde))
    
            for index in range(1, self.observations.shape[0]):
                if np.isnan(self.observations[index]):
                    piX = self.backward_quantity_matrix[index] * self.transition_matrix[discretized_state] 
                else:
                    piX = self.backward_quantity_matrix[index] * self.transition_matrix[discretized_state] * self.emission_matrix[index]
    
                piX /= np.sum(piX)
                discretized_state = np.where(np.random.multinomial(n=1, pvals=piX, size=1)[0]==1)[0][0]
                trajectory[index] = self.X_ronde[discretized_state]
            trajectories_matrix[:,nsim] = trajectory
    
        return trajectories_matrix

    def compute_pointwise_outlier_probabilities(self):
        # Computes marginal pointwise outlier probabilities conditional to observations 
        # and parameters estimations, except the latent variable that is kept uncertain
        self.pointwise_pout = np.ones(self.observations.shape[0]) * np.nan

        # Vectorizing these computations first so that we don't have to repeat them in the next for loop:
        this_partial_emission_vector = np.ones(self.observations.shape[0]) / (self.borne_sup - self.borne_inf)
        this_partial_emission_vector[self.observations_below_LoD] = (self.lod_vect[self.observations_below_LoD] - self.borne_inf) / (self.borne_sup - self.borne_inf)
        this_partial_emission_vector *= self.p_out

        # Treating special case where the investigated observation is gathered at the first timestep:
        # Last index can also be seen as another special case from Bayes standpoint, but the calculation is identical
        # to the general case as backward_vector is equal to 1 at last step anyway
        # This case is treated outside the for loop to maximize efficiency:
        this_forward_vector = 1

        for this_timestep in range(self.observations.shape[0]):
            if this_timestep in self.T_ronde:
                # For every other timestep than 0:
                if this_timestep!=0:
                    this_forward_vector = self.forward_matrix[this_timestep-1]
                            
                this_emission_vector = self.emission_matrix[this_timestep] * np.exp(self.rowMax)[this_timestep, 0]
                this_backward_vector = self.backward_matrix[this_timestep]
                
                num = (this_partial_emission_vector[this_timestep] * np.dot(this_forward_vector, self.transition_matrix) * this_backward_vector).sum() # We need to pay attention not to simply perform vector multiplications here, dot product is required between forward and transition matrixes to accurately perform vectorization
                denom = (this_emission_vector * np.dot(this_forward_vector, self.transition_matrix) * this_backward_vector).sum() # Same observation as the very previous row
                    
                self.pointwise_pout[this_timestep] = (num/denom)