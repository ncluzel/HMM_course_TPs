import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

def get_95CI(signal, alpha=5.0):
    """Computes the (default to 95%) confidence intervals of a sequential signal.

    Parameters
    ----------
    signal : {array-like} of shape (n_samples, n_timesteps).
    The matrix of samples. Each row represents a sample, while each column is associated to a timestep.

    alpha : float.
    1 - CI_range, where CI_range represents the range of the confidence interval.

    Returns
    -------
    CI95_lower : {array} of shape (n_timesteps, ).
    The lower bound of the (1 - alpha) confidence interval.

    CI95_upper : {array} of shape (n_timesteps, ).
    The upper bound of the (1 - alpha) confidence interval.
    """
    CI95_lower = []
    CI95_upper = []

    for timestep in range(signal.shape[1]):
        
        drawn_at_time_t = signal[:,timestep] # Gather the samples at time t
        lower_p = alpha / 2.0 # Computes the lower bound
        lower = np.percentile(drawn_at_time_t, lower_p) # Retrieves the observation at the lower percentile index
        upper_p = (100 - alpha) + (alpha / 2.0) # Computes the upper bound
        upper = np.percentile(drawn_at_time_t, upper_p) # Retrieves the observation at the upper percentile index

        CI95_lower.append(lower)
        CI95_upper.append(upper)
        
    return np.array(CI95_lower), np.array(CI95_upper)


def normal_distribution_pdf(x, loc=0, scale=1): 
    """Computes the normal probability density function (PDF) for each value in the input vector x.
    NB : Empirically much faster than scipy's, probably related to sample size I guess.

    Parameters
    ----------
    x : {array} of shape (n_samples, ).
    The array upon which the PDF is going to be applied.

    loc : float.
    The mean of the normal distribution.

    scale : float.
    The standard deviation of the normal distribution.

    Returns
    -------
    pdf : {array} of shape (n_samples, ).
    The PDF related to the input vector x.
    """
    A = 1 / (scale * np.sqrt(2 * np.pi))
    B = - (1/2) * ((x - loc)/ scale) ** 2
    
    return A * np.exp(B)

def approx_standard_normal_cdf_sw(x, loc=0, scale=1):
    """Computes the normal cumulative distribution function (CDF) for each value in the input vector x,
    using Page's approximation formula.

    Parameters
    ----------
    x : {array} of shape (n_samples, ).
    The array upon which the CDF is going to be applied.

    loc : float.
    The mean of the normal distribution.

    scale : float.
    The standard deviation of the normal distribution.

    Returns
    -------
    cdf : {array} of shape (n_samples, ).
    The CDF related to the input vector x.
    """
    xx = (x - loc) / scale
    return 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (xx + 0.044715 * xx**3)))

def censored_normal_logcdf(lod, eps, latent):
    """Computes the logarithm of the normal cumulative distribution function (CDF) for each observation in the input vector lod,
    based on the assumption that lod|latent \sim \mathcal{N}(latent, eps**2) in pyMC's framework.

    Parameters
    ----------
    lod : {array} of shape (n_steps, ).
    The array of sequential observations. Can contain np.nan values corresponding to unobserved timesteps.

    eps : float.
    The standard deviation of the observations w.r.t the latent variable.

    latent : {array} of shape (n_steps, ).
    The array of the sequential values for the latent variable.

    Returns
    -------
    cdf : {array} of shape (n_steps, ).
    The CDF related to the input vector lod.
    """
    return pm.logcdf(pm.Normal.dist(mu=latent, sigma=eps), lod)

def censored_uniform_logcdf(lod, a, b):
    """Computes the logarithm of the uniform cumulative distribution function (CDF) for each observation in the input vector lod,
    based on the assumption that lod|latent \sim \mathcal{U}(a, b) in pyMC's framework.

    Parameters
    ----------
    lod : {array} of shape (n_steps, ).
    The array of sequential observations. Can contain np.nan values corresponding to unobserved timesteps.

    a : float.
    Lower bound of the uniform distribution.

    b : {array} of shape (n_steps, ).
    Upper bound of the uniform distribution.

    Returns
    -------
    cdf : {array} of shape (n_steps, ).
    The CDF related to the input vector lod.
    """
    return pm.logcdf(pm.Uniform.dist(lower=a, upper=b), lod)


def censored_normal_logcdf_LoQ(loqd, eps, latent):
    """Computes the logarithm of the normal cumulative distribution function (CDF) for each observation in the input vector value,
    based on the assumption that value|latent \sim \mathcal{N}(latent, eps**2) in pyMC's framework.

    Parameters
    ----------
    value : {array} of shape (n_steps, ).
    The array of sequential observations. Can contain np.nan values corresponding to unobserved timesteps.

    eps : float.
    The standard deviation of the observations w.r.t the latent variable.

    latent : {array} of shape (n_steps, ).
    The array of the sequential values for the latent variable.

    Returns
    -------
    cdf : {array} of shape (n_steps, ).
    The CDF related to the input vector value.
    """
    contrib_a = pm.math.exp(censored_normal_logcdf(loqd[:,0], eps, latent))
    contrib_b = pm.math.exp(censored_normal_logcdf(loqd[:,1], eps, latent))
    return pm.math.log(contrib_b - contrib_a)

def censored_uniform_logcdf_LoQ(loqd, a, b):
    """Computes the logarithm of the uniform cumulative distribution function (CDF) for each observation in the input vector value,
    based on the assumption that value|latent \sim \mathcal{U}(a, b) in pyMC's framework.

    Parameters
    ----------
    value : {array} of shape (n_steps, ).
    The array of sequential observations. Can contain np.nan values corresponding to unobserved timesteps.

    a : float.
    Lower bound of the uniform distribution.

    b : {array} of shape (n_steps, ).
    Upper bound of the uniform distribution.

    Returns
    -------
    cdf : {array} of shape (n_steps, ).
    The CDF related to the input vector value.
    """
    contrib_a = pm.math.exp(censored_uniform_logcdf(loqd[:,0], a, b))
    contrib_b = pm.math.exp(censored_uniform_logcdf(loqd[:,1], a, b))
    return pm.math.log(contrib_b - contrib_a)

class SCOU():
    """
    Bayesian implementation of the SCOU algorithm.

    SCOU is an extended Kalman Smoother, taking into consideration
    left-censored values as well as outliers.

    Parameters
    ----------
    p_out_frozen : bool, default=False
        Whether to estimate the a priori outlier probability. If set
        to False, a deterministic value will be used.

    p_out_deterministic : float, default=None
        The value to be used for p_out if p_out_frozen is set to True.

    tuning_iters : int, default=4000
        The number of tuning iterations used for the NUTS MCMC sampler.

    sampling_iters : int, default=2000
        The number of sampling iterations used for the NUTS MCMC sampler.

    nb_chains : int, default=3
        The number of Markov chains used for the NUTS MCMC sampler.

    export_chains : bool, default=False
        Whether to export the chains of parameters in a *.nc file.

    export_name : string, default='default.nc'
        The name of the export file if export_chains is set to True.

    RW_order : {1, 2}, default=1
        The order of the gaussian random walk of the underlying process.

    Attributes
    ----------
    latent_posterior_distribution : array of shape (n_samples, n_steps) 
        The posterior distribution of the latent variable.

    muX : array of shape (n_steps, )
        The average signal of the latent variable, performed over all samples.

    CIU : array of shape (n_steps, )
        The upper bound of the 95% CI of the latent variable.

    CIL : array of shape (n_steps, )
        The lower bound of the 95% CI of the latent variable.

    pointwise_pout : array of shape (n_steps, )
        The posterior probability of each observation to be an outlier.

    See Also
    --------

    Notes
    -----
    This algorithm was tailor-made to meet the expectations of the Obepine research consortium
    during the Covid-19 pandemic in terms of microbiological data processing.

    References
    ----------

    .. [1] M. Courbariaux et al., "A Flexible Smoother Adapted to Censored Data
           With Outliers and Its Application to SARS-CoV-2 Monitoring in Wastewater",
           Frontiers in Applied Mathematics and Statistics, 2022. 
           https://doi.org/10.3389/fams.2022.836349

    """

    def __init__(self, observations,
                 censoring_threshold_lod_vect=np.array([]),
                 censoring_threshold_loq_vect=np.array([]),
                 p_out_frozen=False,
                 p_out_deterministic=None,
                 tuning_iters=4000,
                 sampling_iters=2000,
                 nb_chains=3,
                 export_chains=False,
                 export_name='default.nc',
                 RW_order=1):

        self.observations = observations
        self.censoring_threshold_lod_vect = censoring_threshold_lod_vect
        self.censoring_threshold_loq_vect = censoring_threshold_loq_vect
        self.n_steps = self.observations.shape[0]
        self.rng = np.random.default_rng(666)
        self.tuning_iters = tuning_iters
        self.sampling_iters = sampling_iters
        self.nb_chains = nb_chains
        self.export_name = export_name
        self.export_chains = export_chains
        self.p_out_frozen = p_out_frozen
        self.p_out_deterministic = p_out_deterministic
        self.RW_order = RW_order


    def obs_discrimination(self):
        """Defines the set of observations, whether they are censored or not, which timesteps are not observed
        and the lower and upper bounds of the uniform distribution.
        
        """
        self.unobserved_indexes = np.where(np.isnan(self.observations))[0]

        if len(self.censoring_threshold_loq_vect)>0:
            self.observations_below_LoD = np.where(self.observations<=self.censoring_threshold_lod_vect)[0]
            self.observations_above_LoQ = np.where(self.observations>self.censoring_threshold_loq_vect)[0]
            self.observations_between_LoQD = np.where((self.observations>self.censoring_threshold_lod_vect) & (self.observations<=self.censoring_threshold_loq_vect))[0]
        else:
            self.observations_below_LoD = np.where(self.observations<=self.censoring_threshold_lod_vect)[0]
            self.observations_above_LoQ = np.where(self.observations>self.censoring_threshold_lod_vect)[0]
            self.observations_between_LoQD = np.array([])

        self.T_ronde = np.setdiff1d(np.arange(self.observations.shape[0]), self.unobserved_indexes)
        self.borne_inf, self.borne_sup = np.nanmin(self.observations) - 2*np.nanstd(self.observations), np.nanmax(self.observations) + 2*np.nanstd(self.observations)


    def model_definition(self):
        """Defines the model parameters and observation in pyMC's framework.
        
        """
        self.obs_discrimination()

        with pm.Model() as self.SCOU_model:
            ### ----- Priors definition ----- ###
            sig = pm.InverseGamma('sig', alpha=2, beta=1) # Latent process innovation's drift
            eps = pm.InverseGamma('eps', alpha=2, beta=1) # Observations standard deviation w.r.t latent process
            if self.p_out_frozen:
                p_out = self.p_out_deterministic # Outliers a priori proportion can be frozen to a deterministic value
            else:
                p_out = pm.Beta('p_out', alpha=2, beta=5) # Outliers a priori proportion                

            init_mean = np.nanmean(self.observations)
            init_std = 5
            init_dist = pm.Normal.dist(init_mean, init_std, shape=self.n_steps)
            if self.RW_order==1:
                latent = pm.AR("latent", rho=np.array([1]), sigma=sig, shape=self.n_steps, init_dist=init_dist) # Latent process X[t] defined as (AR(1))
            elif self.RW_order==2:
                latent = pm.AR("latent", rho=np.array([2, -1]), sigma=sig, shape=self.n_steps, init_dist=init_dist) # Latent process X[t] defined as (AR(2))

            ### ----- Uncensored data handling ----- ###
            normal_component = pm.Normal.dist(mu=latent[self.observations_above_LoQ], sigma=eps) # Gaussian component for uncensored data
            outlier_component = pm.Uniform.dist(lower=self.borne_inf, upper=self.borne_sup) # Uniform component for outliers

            ### -----  Uncensored data mixture model definition ----- ###
            obs_uncensored = pm.Mixture(
                'obs_uncensored',
                w=[1 - p_out, p_out],
                comp_dists=[normal_component, outlier_component],
                observed=self.observations[self.observations_above_LoQ]
            )

            ### ----- LoD censored data handling ----- ###
            normal_LoD_component = pm.DensityDist.dist(
                eps, latent[self.observations_below_LoD],
                logp=censored_normal_logcdf,
                logcdf=censored_normal_logcdf,
                class_name="normal_LoD_component"
            )

            outlier_LoD_component = pm.DensityDist.dist(
                self.borne_inf, self.borne_sup, 
                logp=censored_uniform_logcdf,
                logcdf=censored_uniform_logcdf,
                class_name="outlier_LoD_component"
            )

            ### -----  Censored data mixture model definition ----- ###
            obs_LoD = pm.Mixture(
                'obs_LoD',
                w=[1 - p_out, p_out],
                comp_dists=[normal_LoD_component, outlier_LoD_component],
                observed=self.observations[self.observations_below_LoD]
            )

            if len(self.censoring_threshold_loq_vect)>0:
                ### ----- LoQ censored data handling ----- ###
                normal_LoQ_component = pm.DensityDist.dist(
                    eps, latent[self.observations_between_LoQD],
                    logp=censored_normal_logcdf_LoQ,
                    logcdf=censored_normal_logcdf_LoQ,
                    class_name="normal_LoQ_component"
                )

                outlier_LoQ_component = pm.DensityDist.dist(
                    self.borne_inf, self.borne_sup, 
                    logp=censored_uniform_logcdf_LoQ,
                    logcdf=censored_uniform_logcdf_LoQ,
                    class_name="outlier_LoD_component"
                )

                loqd_stacked_obs = np.vstack((self.censoring_threshold_lod_vect[self.observations_between_LoQD],
                                           self.observations[self.observations_between_LoQD])).T

                ### -----  Censored data mixture model definition ----- ###
                obs_LoQ = pm.Mixture(
                    'obs_LoQ',
                    w=[1 - p_out, p_out],
                    comp_dists=[normal_LoQ_component, outlier_LoQ_component],
                    observed=loqd_stacked_obs
                )

    def fit(self):
        """Computes the MCMC estimation of the model parameters using the NUTS sampler.

        """
        self.model_definition()

        # Inférence
        with self.SCOU_model:
            self.SCOU_traces = pm.sample(self.sampling_iters, tune=self.tuning_iters, 
                                    chains=self.nb_chains, 
                                    return_inferencedata=True, 
                                    random_seed=self.rng)

        self.params = ['sig', 'eps'] 
        if not self.p_out_frozen:
            self.params.append('p_out')

        print("Raw summary:")
        print(az.summary(self.SCOU_traces, var_names=self.params))
        self.params_summary = az.summary(self.SCOU_traces, var_names=self.params)

        if self.export_chains:
            self.SCOU_traces.to_netcdf(self.export_name)

    def predict(self, selected_chains):
        """Computes the latent distribution, as well as its mean and 95% confidence intervals and pointwise outlier probabilities
        for a subset of selected chains.

        Parameters
        ----------
        selected_chains : {array} of shape (n_selected_chains, ).
        The array of indexes of the selected chains, ranging from 0 to self.nb_chains.
        """
        self.n_samples = len(selected_chains)*self.sampling_iters
        self.latent_posterior_distribution = self.SCOU_traces['posterior']['latent'].values[selected_chains].reshape(self.n_samples, -1)
        self.muX = self.latent_posterior_distribution.mean(axis=0)
        self.CIL, self.CIU = get_95CI(self.latent_posterior_distribution)
        self.compute_pointwise_outlier_probabilities(selected_chains)

        print("Best chain combination summary:")
        print(az.summary(self.SCOU_traces.sel(chain=selected_chains), var_names=self.params))

    def compute_pointwise_outlier_probabilities(self, selected_chains):
        """Computes the pointwise outlier probabilities for a subset of selected chains.

        Parameters
        ----------
        selected_chains : {array} of shape (n_selected_chains, ).
        The array of indexes of the selected chains, ranging from 0 to self.nb_chains.
        """
        nb_draws = self.SCOU_traces['posterior']['eps'].values[selected_chains].shape[1] * len(selected_chains)

        self.pointwise_pout = np.ones(self.observations.shape[0]) * np.nan
        self.pointwise_pout_dist = np.ones((self.observations.shape[0], nb_draws)) * np.nan

        # Vectorizing these computations first so that we don't have to repeat them in the next for loop:
        this_partial_emission_vector = np.ones(self.observations.shape[0]) * (1/(self.borne_sup - self.borne_inf))
        this_partial_emission_vector[self.observations_below_LoD] = ((self.censoring_threshold_lod_vect[self.observations_below_LoD] - self.borne_inf)/(self.borne_sup - self.borne_inf))
        if len(self.censoring_threshold_loq_vect)>0:
            this_partial_emission_vector[self.observations_between_LoQD] = ((self.censoring_threshold_loq_vect[self.observations_between_LoQD] - self.censoring_threshold_lod_vect[self.observations_between_LoQD])/(self.borne_sup - self.borne_inf))

        for this_timestep in self.T_ronde:     
            xhat_t = self.observations[this_timestep]
            x_t = self.SCOU_traces['posterior']['latent'].values[selected_chains].reshape(len(selected_chains)*self.sampling_iters, -1)[:, this_timestep]
            this_epsilon = self.SCOU_traces['posterior']['eps'].values[selected_chains].reshape(len(selected_chains)*self.sampling_iters, )[:,]

            if self.p_out_frozen:
                this_pout = self.p_out_deterministic
            else:
                this_pout = self.SCOU_traces['posterior']['p_out'].values[selected_chains].reshape(len(selected_chains)*self.sampling_iters, )[:,]

            if this_timestep in self.observations_below_LoD:
                num = this_pout * this_partial_emission_vector[this_timestep]
                denom_not_outlier = (1-this_pout) * approx_standard_normal_cdf_sw(xhat_t, x_t, this_epsilon)
                denom = denom_not_outlier + num

            elif this_timestep in self.observations_between_LoQD:
                num = this_pout * this_partial_emission_vector[this_timestep]
                cdf_diff = approx_standard_normal_cdf_sw(xhat_t, x_t, this_epsilon) - approx_standard_normal_cdf_sw(self.censoring_threshold_lod_vect[this_timestep], x_t, this_epsilon)
                denom_not_outlier = (1-this_pout) * cdf_diff
                denom = denom_not_outlier + num
            
            elif this_timestep in self.observations_above_LoQ:    
                num = this_pout * this_partial_emission_vector[this_timestep]
                denom_not_outlier = (1-this_pout) * normal_distribution_pdf(xhat_t, x_t, this_epsilon)
                denom = denom_not_outlier + num

            num = np.array(num)
            denom = np.array(denom)
            
            self.pointwise_pout_dist[this_timestep] = (num/denom)
            self.pointwise_pout[this_timestep] = np.mean(self.pointwise_pout_dist[this_timestep])

    def visualize_latents(self, selected_chains):
        """Plots the mean of the distributions of the latent variable for each chain on a first figure.
        Plots the same distribution only for a subset of selected chains on a second figure.

        Parameters
        ----------
        selected_chains : {array} of shape (n_selected_chains, ).
        The array of indexes of the selected chains, ranging from 0 to self.nb_chains.
        """
        plt.figure()
        for i in range(self.nb_chains):
            plt.plot(self.SCOU_traces['posterior']['latent'][i].mean(axis=0), label=i)

        plt.title('Raw chains')
        plt.legend()
        plt.show()

        plt.figure()
        for i in selected_chains:
            plt.plot(self.SCOU_traces['posterior']['latent'][i].mean(axis=0), label=i)

        plt.title('Optimized chains')
        plt.legend()
        plt.show()