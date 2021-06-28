import helper

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import  StrMethodFormatter

import skmultiflow as sm

import shap
import lime
from fires import FIRES

from sklearn.base import copy
from contextlib import redirect_stdout
from pathlib import Path

import time

from scipy.linalg import solve
from scipy.stats import invwishart



class XEvaluator:
    '''
    Evaluates the performance of the classifier on the given data stream, generates explanations according to the 
    defined configuration and evaluates the robustness of the generated explanations. Uses prequential evaluation 
    method.
    '''
    
    def __init__(self, stream, classifier, explainers, explainer_weights=None, ensemble=False, ensemble_fires=None, 
                 param_samples=1, expl_nsamples=None, pretrain_size=100, baseline_window=100, max_batches=5_000, 
                 random_state=500, agg_fun='sample_average', plot=False, acc_win=250, explanation_ground_truth=True, 
                 top_K=10, clf_name=None, expl_qual_measure='distance', results_dir=None, fires_win_size=1, ann=False,
                 use_proba=False, custom_param_samples=False, custom_baseline=False, stream_name=None, custom_data=None):
        '''
        Initializes XEvaluator configured with the given parameters.
        
        Args:
            stream (Stream): Stream created with 'skmultiflow'
            classifier (Classifier): Classifier which implements `predict` and possibly `predict_proba` functions like 
                in 'sklearn'.
            explainers (str / list): String representing an explainer or a list of explainers. Possible explainers:
                - 'shap': SHAP explainer which calculates approximate SHAP values using permutation algorithm.
                - 'lime': LIME explainer
                - 'fires': FIRES as explainer
                - 'custom': Custom explainer
            explainer_weights (list, optional): Weights given to explainers. Used if a list of explainers is given.
            ensemble (bool, optional): Whether to use ensemble of the given explainers. Works only if 'shap' or 'lime' 
                were given as explainers.
            ensemble_fires (FIRES, optional): If given uses it as the FIRES model for keeping track of the model 
                distribution used in ensemble explainer models.
            param_samples (int, optional): How many model parameters to sample for ensemble classifier?
            expl_nsamples (int, optional): Number of samples for explanation of an instance. For example, number of
                feature coalitions in SHAP or number of samples around the explained instance in LIME. SHAP has a lower
                limit of 2 * num_features + 1. 
            pretrain_size (int, optional): Number of samples to pretrain on. Should be < size of the data stream. 
                Should be > `baseline_window`.
            baseline_window (int, optional): Number of previously encountered samples used as the baseline in 'shap' 
                and 'lime' explainers. Should be < `pretrain_size`. If less, XEvaluator starts with `pretrain_size` 
                samples used for the baseline and adds the following samples to baseline until it reaches the 
                `baseline_window` size which it then maintains.
            max_batches (None, optional): Maximum number of batches from the stream to look at.
            random_state (int, optional): Used as the seed for all parts of the evaluator that accept the seed.
            agg_fun (str, optional): Manner in which `param_samples` are aggregated. Possible values:
                - 'sample_average': Takes and average of the sampled model parameters.
                - 'distrib_mean': Uses mean of the FIRES distribution that keeps track of model parameters
            plot (bool, optional): Whether to show plots during the evaluation.
            acc_win (int, optional): Window used for calculation of accuracy.
            clf_name (str, optional): Name of the used classifier. Used for documentation of results.
            explanation_ground_truth (bool, optional): Whether the coefficients of the given classifier can be used as 
                ground truth attributions. Classifier should contain attribute `coef_` (like `sklearn.linear_model` 
                module) which contains the attributions.
            top_K (int, optional): Number of top K features to show for in plots.
            expl_qual_measure (str, optional): Which measure to use for local quality of explanations.
            results_dir (None, optional): Directory in which to store the results of the current run. Appends to the 
                data if it already exists.
            fires_win_size (int, optional): Number of samples to use in the window for FIRES if it is used as an 
                explainer.
            ann (bool, optional): Whether used model is an artificial neural network. Currently assumes that it only
                has fully connected layers. Assumed that it has an attribute `.coefs_` like in `MLPClassifier` in 
                sklearn.
            use_proba (bool, optional): Whether to use `.predict_proba()` method of the classifier in calculation of 
                explanations.
            custom_param_samples (bool, optional): Whether the selected custom explainer needs classifier 
                model parameter samples. Relevant only if custom explainer is used.
            custom_baseline (bool, optional): Whether the selected custom explainer needs baseline samples. Relevant 
                only if custom explainer is used.
            custom_data (not-specified; e.g. dict, optional): Custom data for the custom explainer. For example, can
                contain initialization of explainer parameters which can be updated for every batch in `evaluate()`.
        '''
        self.stream = stream
        self.classifier = classifier
        self.has_proba = hasattr(self.classifier, 'predict_proba') and callable(self.classifier.predict_proba)
        self.explainers = explainers if isinstance(explainers, list) else [explainers]
        
        if ensemble_fires is None:
            fires_model = FIRES(n_total_ftr=stream.n_features,
                                target_values=stream.target_values,
                                mu_init=0,
                                sigma_init=1,
                                lr_mu=1,
                                lr_sigma=1,
                                scale_weights=False)
            self.ensemble_fires = fires_model

        self.ensemble = ensemble
        if ensemble:
            self.ensemble_clfs = [copy.deepcopy(self.classifier) for _ in range(param_samples)]

        self.param_samples = param_samples
        self.agg_fun = agg_fun
        self.expl_nsamples = 2*stream.n_features + 1 if expl_nsamples is None else expl_nsamples

        if explainer_weights is None:
            self.explainer_weights = len(self.explainers)*[1/len(self.explainers)]

        if use_proba:
            assert self.has_proba, 'Cannot `use_proba` is classifer has no `predict_proba()` method.'
        self.use_proba = use_proba

        self.rng = np.random.default_rng(random_state)
        self.curr_batch = 0
        self.correct = []

        if 'shap' in explainers or 'lime' in explainers or custom_baseline:
            assert (baseline_window > 1), 'Selected explainer needs a baseline.'

        self.baseline_window = baseline_window
        assert (pretrain_size > 0), 'Pretrain size has to be > 0 because the model has to initialize coefficients.'
        self.pretrain_size = pretrain_size
        self.max_batches = max_batches
        
        self.global_attr = None
        self.top_K = top_K
        self.plot = plot
        self.plotting_initialized = False
        self.acc_win = acc_win
        self.fires_win_size = fires_win_size
        self.ann = ann

        self.explanation_ground_truth = explanation_ground_truth
        self.expl_qual_measure = expl_qual_measure

        self.clf_name = clf_name
        self.stream_name = stream_name

        self.results_dir = results_dir

        self.custom_param_samples = custom_param_samples
        self.custom_baseline = custom_baseline
        self.custom_data = custom_data

        self.random_state = random_state
        np.random.seed(random_state)
    

    def pretrain_classifier(self):
        '''
        Pretrains a classifier using `pretrain_size` examples from the stream.
        
        Returns:
            (np.ndarray, np.ndarray, np.ndarray[, np.ndarray]): Returns `X`, `y`, `y_preds` and if classifier contains 
                `predict_proba` function `y_probas` where for first `pretrain_size` samples in the data stream used to 
                pretrain the classifier `X` contains data points, `y` labels, `y_preds` predictions made by the 
                classifier and `y_probas` probabilities of belonging to a certain class. 
        '''
        X, y = self.stream.next_sample(batch_size=self.pretrain_size)
        if self.ensemble:
            for classifier in self.ensemble_clfs:
                classifier.partial_fit(X, y, self.stream.target_values)
                if self.ann:
                    classifier.coef_ = np.mean(helper.get_gt_attrib_ann(classifier, X[0:1]).T, axis=0, keepdims=True)
        
        # Classifier whose attributions should be explained should always be fitted
        self.classifier.partial_fit(X, y, self.stream.target_values)

        # Immediately assign attributions to classifier
        if self.ann:
            self.classifier.coef_ = np.mean(helper.get_gt_attrib_ann(self.classifier, X[0:1]).T, axis=0, keepdims=True)
        
        self.global_attr = np.empty((1, X.shape[1]))

        y_preds = self.classifier.predict(X)

        if self.use_proba:
            y_probas = self.classifier.predict_proba(X)
            return X, y, y_preds, y_probas[:,1]

        return X, y, y_preds
    
    
    def sample_model_params(self, X, y):
        '''
        Samples model parameters using FIRES and the given parameters.

        Should be called after 'pretrain_classifier()' because `coef_`
        only instantiated after `partial_fit()` method is called.
        
        Args:
            X (np.ndarray): data points
            y (np.ndarray): labels
        '''
        if self.ensemble_fires and self.agg_fun == 'distrib_mean':
            self.ensemble_fires.weigh_features(X, y)
            return self.ensemble_fires.mu[np.newaxis,:]

        elif self.ensemble_fires and self.param_samples > 1:
            # Updates model distribution parameters
            self.ensemble_fires.weigh_features(X, y)

            size = (self.param_samples, self.ensemble_fires.mu.shape[0])

            if self.ensemble:
                return self.rng.normal(self.ensemble_fires.mu, self.ensemble_fires.sigma, size)
            else:
                return np.mean(self.rng.normal(self.ensemble_fires.mu, self.ensemble_fires.sigma, size), axis=0)[np.newaxis,:]

        else:
            return self.classifier.coef_


    def _attrib_shap(self, X, clf, baseline_sample):
        '''Calculates attributions using SHAP explainer'''
        if self.use_proba:
            clf_output = clf.predict_proba
        else:
            clf_output = clf.predict
        explainer = shap.Explainer(clf_output, baseline_sample, algorithm='permutation')
        return explainer(X, max_evals=self.expl_nsamples).values[0]


    def _attrib_lime(self, X, clf, baseline_sample, random_state=None):
        '''Calculates attributions using LIME explainer'''
        if random_state is None:
            random_state = self.random_state

        tmp_attrib_values = np.empty_like(X)
        explainer = lime.lime_tabular.LimeTabularExplainer(baseline_sample, mode='classification', random_state=random_state, 
                                                           feature_names=list(range(X.shape[-1])), discretize_continuous=False)

        if self.use_proba:
            clf_output = clf.predict_proba
        else:
            def pred_as_prob(X):
                is0 = (clf.predict(X)==0)
                is1 = np.logical_not(is0)
                return np.column_stack((is0.astype(float), is1.astype(float)))
            clf_output = pred_as_prob
        for j in range(X.shape[0]):
            lime_explanations = explainer.explain_instance(X[j], clf_output, 
                                                           num_features=X.shape[-1], 
                                                           num_samples=self.expl_nsamples).as_list()
            
            tmp_attrib_values[j] = np.array([x[1] for x in sorted(lime_explanations, key=lambda x: x[0])])

        return np.mean(tmp_attrib_values, axis=0)


    # Example definition for a custom explainer
    def _attrib_weighted_gauss(self, X, y, expl_clf, baseline_sample, mu_prior, sigma_prior=None, lik_sigma_prior=None, 
                               prior_explanations=None, normalize_instances=True, lik_sigma=0.1):
        '''
        - possibility to mix shap or lime values for the current data point in the prior
        - likelihood sigma used as a parameter
        - not adapted for different batch sizes
        - code to infer the likelihood sigma with inverse wishart prior commented out
        '''
        y = (-1)**(1-y)

        # Dimension of attributions with intersect
        dim = X.shape[-1]+1

        # Use non-informative prior on the likelihood cov
        # if lik_sigma_prior is None:
        #     lik_sigma = invwishart.rvs(lik_sigma_prior)

        if sigma_prior is None:
            sigma_prior = 0.1*np.identity(dim)
 

        x = np.ones((dim, 1))
        x[:dim-1,0] = X.flatten()

        if normalize_instances:
            x = x / np.sum(np.abs(x))

        if prior_explanations is not None:
            attribs = np.zeros((dim, 1))

            if prior_explanations =='shap':
                attribs[:dim-1,0] = self._attrib_shap(X, expl_clf, baseline_sample)
            elif prior_explanations == 'lime':
                attribs[:dim-1,0] = self._attrib_lime(X, expl_clf, baseline_sample)

            mu_prior = (mu_prior + attribs) / 2
            mu_prior *= 2


        # # Take the mode
        # lik_sigma = lik_sigma_prior[1] / (lik_sigma_prior[0] + 1 + 1)

        gram = x.T @ sigma_prior @ x + np.identity(dim) * lik_sigma
        gram_inv = solve(gram + np.identity(X.shape[-1]+1)*1e-7, np.identity(X.shape[-1]+1), assume_a='sym')

        out = np.tanh(x.T @ mu_prior)
        residual = (y - out)

        # lik_sigma_post = (lik_sigma_prior[0] + 1, lik_sigma_prior[1] + (y - out)*(y-out))
        sigma_post = sigma_prior - x.T @ sigma_prior @ gram_inv @ sigma_prior @ x
        mu_post = mu_prior +  residual * (sigma_post @ gram_inv @ x)


        # return mu_post, sigma_post, lik_sigma_post
        return mu_post, sigma_post, None


    def init_plotting(self):
        '''
        Initializes figures and axes.
        
        Returns:
            figure, axes: Returns figure containing subplots and following axes:
                - `ax_pred`: Plots accuracy of the classifier that is explained
                - `ax_expl_stability`: Plots stability of explanations, i.e. accumulated distances between explanations 
                    at time t and time t-1.
                - `ax_expl_qual`: Local quality of explanations, i.e. accumulated distances between explanations and 
                    linear model weights (of classifier that is explained) at time t. `None` if no ground truth.
                - `ax_expl_glob`: Cumulative explainer feature attributions of top K features. Represents global 
                    attribution.
                - `ax_truth`: Cumulative explained classifier's linear model weights. 
        '''
        self.drawn_glob_features = {}
        self.drawn_truth_features = {}

        fig = plt.figure(constrained_layout=True)
        gs = fig.add_gridspec(nrows=3, ncols=2)

        ax_pred = fig.add_subplot(gs[0,0])
        ax_expl_stability = fig.add_subplot(gs[0,1])
        ax_expl_glob = fig.add_subplot(gs[1,:])
        ax_truth = fig.add_subplot(gs[2,:])

        if self.explanation_ground_truth:
            ax_expl_qual = ax_expl_stability.twinx()

            ax_expl_qual.set_ylabel('attribution\ndiscrepancy', fontsize='xx-small', rotation=-90, labelpad=18)
            ax_expl_qual.yaxis.set_tick_params(labelsize='xx-small')
            ax_expl_qual.yaxis.set_major_formatter(StrMethodFormatter('{x:5.0f}'))

            axes = (ax_pred, ax_expl_stability, ax_expl_qual, ax_expl_glob, ax_truth)   

        else:
            axes = (ax_pred, ax_expl_stability, None, ax_expl_glob, ax_truth)   

        for ax in axes[:-1]:
            plt.setp(ax.get_xticklabels(), visible=False)

        ax_truth.set_xlabel('time t')

        for ax in axes:
            ax.yaxis.LABELPAD = 25
            ax.yaxis.set_tick_params(labelsize='xx-small')
            ax.yaxis.set_major_formatter(StrMethodFormatter('{x:5.0f}'))
        ax_pred.yaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))

        ax_expl_glob.set_ylabel('global explanations', fontsize='xx-small')
        ax_truth.set_ylabel('ground truth', fontsize='xx-small')
        ax_pred.set_ylabel('accuracy', fontsize='xx-small')
        ax_expl_stability.set_ylabel('stability', fontsize='xx-small')

        time = np.arange(self.curr_batch+1 if self.curr_batch+1 <= self.max_batches else self.max_batches)

        ax_pred.plot(time, self.accuracy[:self.curr_batch+1], c='b', label='accuracy')
        ax_pred.yaxis.grid()
        ax_pred.legend(fontsize='xx-small')

        ax_expl_stability.plot(time, np.cumsum(self.stability[:self.curr_batch+1]), c='r', label='stability')
        ax_expl_stability.legend(loc='upper left', fontsize='xx-small')
        if ax_expl_qual is not None:
            ax_expl_qual.plot(time, np.cumsum(self.expl_qual[:self.curr_batch+1]), c='g', label='explanation distance', 
                              alpha=0.6)
            ax_expl_qual.legend(loc='lower right', fontsize='xx-small')

        for i in range(len(self.top_K_features)):
            line = ax_expl_glob.plot(time, self.cum_ftr_attributions[:self.curr_batch+1, self.top_K_features[i]], 
                                     label=f'feature {self.top_K_features[i]}', lw=0.7)[0]
            self.drawn_glob_features[self.top_K_features[i]] = line
            
            if self.explanation_ground_truth:
                line_truth = ax_truth.plot(time, self.truth[:self.curr_batch+1, self.top_K_features_truth[i]], 
                                           label=f'feature {self.top_K_features_truth[i]}', lw=0.7)[0]
                self.drawn_truth_features[self.top_K_features_truth[i]] = line_truth


        ax_expl_glob.legend(handles=[self.drawn_glob_features[ftr] for ftr in self.top_K_features], 
                            bbox_to_anchor=(1.01,1.0), 
                            borderaxespad=0, 
                            fontsize='xx-small')
        ax_truth.legend(handles=[self.drawn_truth_features[ftr] for ftr in self.top_K_features_truth], 
                        bbox_to_anchor=(1.01,1.0), 
                        borderaxespad=0, 
                        fontsize='xx-small')

        return (fig, *axes)


    def update_figure(self, ax_expl_stability, ax_pred, ax_expl_glob, ax_truth, ax_expl_qual=None):
        time = np.arange(self.curr_batch+1)

        ax_expl_stability.get_lines()[0].set_data(time, np.cumsum(self.stability[:self.curr_batch+1]))
        ax_expl_stability.relim()
        ax_expl_stability.autoscale_view()
        if ax_expl_qual is not None:
            ax_expl_qual.get_lines()[0].set_data(time, np.cumsum(self.expl_qual[:self.curr_batch+1]))
            ax_expl_qual.relim()
            ax_expl_qual.autoscale_view()

        ax_pred.get_lines()[0].set_data(time, self.accuracy[:self.curr_batch+1])
        ax_pred.relim()
        ax_pred.autoscale_view()

        for i in range(len(self.top_K_features)):
            if self.top_K_features[i] in self.drawn_glob_features:
                self.drawn_glob_features[self.top_K_features[i]].set_data(time, 
                                                self.cum_ftr_attributions[:self.curr_batch+1, self.top_K_features[i]])
            else:
                line = ax_expl_glob.plot(time, self.cum_ftr_attributions[:self.curr_batch+1, self.top_K_features[i]], 
                                         label=f'feature {self.top_K_features[i]}')[0]
                self.drawn_glob_features[self.top_K_features[i]] = line
            
            if self.explanation_ground_truth:
                if self.top_K_features_truth[i] in self.drawn_truth_features:
                    self.drawn_truth_features[self.top_K_features_truth[i]].set_data(time, 
                                                self.truth[:self.curr_batch+1, self.top_K_features_truth[i]])
                else:
                    line_truth = ax_truth.plot(time, self.truth[:self.curr_batch+1, self.top_K_features_truth[i]], 
                                               label=f'feature {self.top_K_features_truth[i]}')[0]
                    self.drawn_truth_features[self.top_K_features_truth[i]] = line_truth


        ax_expl_glob.legend(handles=[self.drawn_glob_features[ftr] for ftr in self.top_K_features], 
                            bbox_to_anchor=(1.01,1.0), 
                            borderaxespad=0, 
                            fontsize='xx-small')
        ax_truth.legend(handles=[self.drawn_truth_features[ftr] for ftr in self.top_K_features_truth], 
                        bbox_to_anchor=(1.01,1.0), 
                        borderaxespad=0, 
                        fontsize='xx-small')

        ax_expl_glob.relim()
        ax_expl_glob.autoscale_view()
        ax_truth.relim()
        ax_truth.autoscale_view()

        
    def evaluate(self, stream=None, batch_size=1):
        '''
        Evaluates explanations and stores results with a plot in `results_dir`.
        
        Args:
            stream (Stream, optional): `skmultiflow.data.Stream` on which explanations are evaluated. If set, used 
                instead of the stream defined at initialization.
            batch_size (int, optional): Number of samples to use at every time step.
        
        Returns:
            tuple: Returns evaluation information:
                - `clf_name`: name of the classifier that was explained
                - `stream_name`: name of the used stream / dataset
                - `accuracy`: accuracy at last time step
                - `explainer info`: Information about explainer (name, weights and whether ensemble was used)
                - `stability`: stability at last time step
                - `timing`: Average time needed for explanation per time step.        
        '''

        if stream is not None:
            self.stream = stream

        
        
        self.cum_ftr_attributions = np.zeros((self.max_batches, self.stream.n_features))

        pos_class_samples = 0
        
        max_samples = self.max_batches * batch_size + self.pretrain_size
        seen_samples = np.zeros((max_samples,self.stream.n_features))

        if self.use_proba:
            seen_probas = np.zeros(max_samples)
            pretrain_sample, pretrain_y, pretrain_preds, pretrain_probas = self.pretrain_classifier()
            seen_probas[:self.pretrain_size] = pretrain_probas
        else:
            seen_preds = np.zeros(max_samples)
            pretrain_sample, pretrain_y, pretrain_preds = self.pretrain_classifier()
            seen_preds[:self.pretrain_size] = pretrain_preds

        seen_samples[:self.pretrain_size] = pretrain_sample
        
        prev_attrib_normalized = 0

        self.accuracy = np.zeros(self.max_batches)
        self.stability = np.zeros(self.max_batches)
        timings = np.zeros(self.max_batches)
        if self.explanation_ground_truth:
            self.expl_qual = np.zeros(self.max_batches)
            self.truth = np.zeros((self.max_batches, self.stream.n_features))
            self.coef_sum = np.zeros((1, self.stream.n_features))


        while self.curr_batch < self.max_batches and self.stream.has_more_samples():
            X, y = self.stream.next_sample(batch_size)
            pos_class_samples += (y == 1)

            # Prediction
            predictions = self.classifier.predict(X)
            if self.use_proba:
                probas = self.classifier.predict_proba(X)

            self.correct.extend(predictions == y)

            # Sampling model parameters w.r.t. prediction
            param_sample = self.sample_model_params(X, predictions)
            
            curr_sample = self.pretrain_size+self.curr_batch*batch_size+X.shape[0]
            seen_samples[curr_sample-X.shape[0]:curr_sample] = X
            if self.use_proba:
                seen_probas[curr_sample-X.shape[0]:curr_sample] = probas[:,1]
            else:
                seen_preds[curr_sample-X.shape[0]:curr_sample] = predictions
            
            acc_win_lend = curr_sample - self.acc_win if curr_sample > self.acc_win else 0
            curr_sample_no_pret = self.curr_batch*batch_size+X.shape[0]
            self.accuracy[self.curr_batch] = np.mean(self.correct[acc_win_lend:curr_sample_no_pret])
            
            # Explanation
            attrib_values = []
            timing_start = time.perf_counter()
            if 'fires' in self.explainers:
                # Feature weighting using predictions
                lend = curr_sample-self.fires_win_size
                if self.use_proba:
                    ftr_weights = self.ensemble_fires.weigh_features(seen_samples[lend if lend > 0 else 0:curr_sample], 
                                                                     seen_probas[lend if lend > 0 else 0:curr_sample])
                else:
                    ftr_weights = self.ensemble_fires.weigh_features(seen_samples[lend if lend > 0 else 0:curr_sample], 
                                                                     seen_preds[lend if lend > 0 else 0:curr_sample])

                attrib_values.append(ftr_weights)

            if self.custom_baseline or 'shap' in self.explainers or 'lime' in self.explainers:
                lend = curr_sample-self.baseline_window
                lend = lend if lend > 0 else 0

                baseline_sample = seen_samples[lend:curr_sample]

                if self.ensemble:
                    predictions = np.empty(len(self.ensemble_clfs))
                    attrib_values_arr = []
                    for i in range(len(self.ensemble_clfs)):
                        self.ensemble_clfs[i].coef_ = param_sample[i:i+1,:]

                        if 'shap' in self.explainers:
                            attrib_values_arr.append(self._attrib_shap(X, self.ensemble_clfs[i], baseline_sample))
                        elif 'lime' in self.explainers:
                            attrib_values_arr.append(self._attrib_lime(X, self.ensemble_clfs[i], baseline_sample))

                    attrib_values.append(np.mean(attrib_values_arr, axis=0))
                else:
                    expl_clf = copy.deepcopy(self.classifier)
                    expl_clf.coef_ = param_sample

                    if 'shap' in self.explainers:
                        attrib_values.append(self._attrib_shap(X, expl_clf, baseline_sample))
                    elif 'lime' in self.explainers:
                        attrib_values.append(self._attrib_lime(X, expl_clf, baseline_sample))
                    # Custom Explainer example with a baseline
                    else:
                        self.custom_data['priors'] = self._attrib_weighted_gauss(X, y, expl_clf, baseline_sample, 
                                                                                 self.custom_data['priors'][0], 
                                                                                 self.custom_data['priors'][1],
                                                                                 self.custom_data['priors'][2])
                        attrib_values.append(self.custom_data['priors'][0].flatten()[:-1])

            # Add custom explainer with no baseline


            timing_stop = time.perf_counter()
            timings[self.curr_batch] = timing_stop - timing_start

            # Weighing of explainers
            for i in range(len(attrib_values)):
                av_norm = np.linalg.norm(attrib_values[i], ord=1)
                attrib_values[i] = (attrib_values[i] / av_norm) if av_norm > 0.001 else np.zeros_like(attrib_values[i])

            attrib_values = np.average(attrib_values, weights=self.explainer_weights, axis=0)[np.newaxis,:]

            # Stability and local quality calculation
            attrib_mean = np.mean(attrib_values, axis=0)
            attrib_norm = np.linalg.norm(attrib_mean, ord=1)
            attrib_normalized = attrib_mean / attrib_norm if attrib_norm != 0 else 0
            coefs_norm = np.linalg.norm(self.classifier.coef_[0], ord=1)
            coefs_normalized = self.classifier.coef_[0] / coefs_norm if coefs_norm != 0 else 0

            self.expl_qual[self.curr_batch] = np.sum(np.abs(coefs_normalized - attrib_normalized))
            self.stability[self.curr_batch] = np.sum(np.abs(attrib_normalized - prev_attrib_normalized))

            assert np.abs(np.sum(np.abs(attrib_normalized)) - 1) < 0.001 or np.all(attrib_normalized) == 0, f'Absolute sum of normalized attributions is {np.sum(np.abs(attrib_normalized))} != 1'
            assert np.abs(np.sum(np.abs(coefs_normalized)) - 1) < 0.001 or np.all(coefs_normalized) == 0, f'Absolute sum of normalized coefs is {np.sum(np.abs(coefs_normalized))} != 1'

            prev_attrib_normalized = attrib_normalized
            
            # Global attributions and ground truth
            self.global_attr += np.sum(np.abs(attrib_values), axis=0, keepdims=True)
            self.cum_ftr_attributions[self.curr_batch] = self.global_attr

            if self.explanation_ground_truth:
                self.coef_sum += np.sum(np.abs(self.classifier.coef_), axis=0, keepdims=True)
                self.truth[self.curr_batch] = self.coef_sum
        
            self.top_K_features = np.flip(np.argsort(self.global_attr, axis=None)[-self.top_K:])
            self.top_K_features.dtype = int
            self.top_K_features_truth = np.flip(np.argsort(self.coef_sum, axis=None)[-self.top_K:])
            self.top_K_features_truth.dtype = int

            # Print every n-th batch
            if self.curr_batch % 10 == 0:
                print(f'{self.curr_batch:5}/{self.max_batches}; Acc: {self.accuracy[self.curr_batch]:4.2f}; ',
                      f'Stability: {self.stability[self.curr_batch]:4.2f}; ',
                      f'ExplDist: {self.expl_qual[self.curr_batch]:4.2f}; Time: {timings[self.curr_batch]:6.5f}', end='\r')

            # Plotting
            if self.plot:
                if not self.plotting_initialized:
                    fig, ax_pred, ax_expl_stability, ax_expl_qual, ax_expl_glob, ax_truth = self.init_plotting()
                    self.plotting_initialized = True
                else:
                    if self.curr_batch % 200 == 0:
                        self.update_figure(ax_expl_stability, ax_pred, ax_expl_glob, ax_truth, ax_expl_qual)

                        plt.pause(0.005)
            
            self.curr_batch += 1
            
            # Train classifier that is explained            
            self.classifier.partial_fit(X, y)

            if self.ann:
                self.classifier.coef_ = np.mean(helper.get_gt_attrib_ann(self.classifier, X).T, axis=0, keepdims=True)

        
        # Store results
        if self.results_dir:
            # path = (Path.cwd() / 'results' / self.results_dir).resolve()
            path = Path(self.results_dir)
            path.mkdir(parents=True, exist_ok=True)
        else:
            path = Path.cwd()

        np.savetxt(
            (path / f'stability_{self.stream_name}_{self.max_batches}_{self.param_samples}\
_{self.expl_nsamples}_{self.ensemble}_{self.clf_name}_{self.expl_qual_measure}').resolve(), 
            np.cumsum(self.stability)
        )
        np.savetxt(
            (path / f'accuracy_{self.stream_name}_{self.max_batches}_{self.param_samples}\
_{self.expl_nsamples}_{self.ensemble}_{self.clf_name}_{self.expl_qual_measure}').resolve(), 
            self.accuracy
        )
        np.savetxt(
            (path / f'explQual_{self.stream_name}_{self.max_batches}_{self.param_samples}\
_{self.expl_nsamples}_{self.ensemble}_{self.clf_name}_{self.expl_qual_measure}').resolve(), 
            np.cumsum(self.expl_qual)
        )
        np.savetxt(
            (path / f'timings_{self.stream_name}_{self.max_batches}_{self.param_samples}\
_{self.expl_nsamples}_{self.ensemble}_{self.clf_name}_{self.expl_qual_measure}').resolve(), 
            np.cumsum(timings)
        )

        with open((path / f'results.txt').resolve(), 'a+') as f:
            with redirect_stdout(f):
                print(f'EXPLAINERS: {self.explainers}')
                print(f'EXPLAINER WEIGHTS: {self.explainer_weights}')
                print(f'accuracy_{self.stream_name}_{self.max_batches}_{self.param_samples}_{self.expl_nsamples}\
_{self.ensemble}_{self.clf_name}_{self.expl_qual_measure}')
                print(f'\t{self.accuracy[-1]}')
                print(f'stability_{self.stream_name}_{self.max_batches}_{self.param_samples}_{self.expl_nsamples}\
_{self.ensemble}_{self.clf_name}_{self.expl_qual_measure}')
                print(f'\t{np.cumsum(self.stability)[-1]}')
                print(f'explQual_{self.stream_name}_{self.max_batches}_{self.param_samples}_{self.expl_nsamples}\
_{self.ensemble}_{self.clf_name}_{self.expl_qual_measure}')
                print(f'\t{np.cumsum(self.expl_qual)[-1]}')
                print(f'timings_{self.stream_name}_{self.max_batches}_{self.param_samples}_{self.expl_nsamples}\
_{self.ensemble}_{self.clf_name}_{self.expl_qual_measure}')
                print(f'\t{np.cumsum(timings)[-1]}')

        if not self.plot:
            fig, ax_pred, ax_expl_stability, ax_expl_qual, ax_expl_glob, ax_truth = self.init_plotting()

        fig.savefig(
            (path / f'{self.explainers}_{self.explainer_weights}_{self.stream_name}\
_{self.max_batches}_{self.param_samples}_{self.expl_nsamples}_{self.ensemble}_{self.clf_name}\
_{self.expl_qual_measure}.png').resolve(), 
            bbox_inches='tight'
        )
        self.stream.restart()

        # Compute Top-K Accuracy and distance
        presence = [ftr in self.top_K_features_truth for ftr in self.top_K_features]
        top_K_acc = sum(presence)
        top_K_distance = sum((self.top_K - list(self.top_K_features_truth).index(self.top_K_features[pos])) / 
                              (1 + abs(pos - list(self.top_K_features_truth).index(self.top_K_features[pos])))
                              for pos in range(len(self.top_K_features)) if presence[pos])
   
        return (self.clf_name, 
                self.stream_name, 
                self.accuracy[-1], 
                str(self.explainers)+str(self.explainer_weights)+f'Ensemble={self.ensemble}', 
                np.cumsum(self.stability)[-1], 
                np.cumsum(self.expl_qual)[-1], 
                top_K_acc,
                top_K_distance,
                (1_000*np.sum(timings)) / self.max_batches)



