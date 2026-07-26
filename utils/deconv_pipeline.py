import mne

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns

import math
import re
from collections import namedtuple
from typing import Callable

from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.linear_model import Ridge

from scipy.stats import ttest_1samp, mode
from statsmodels.stats.multitest import fdrcorrection

from collections import OrderedDict

RANDOM_STATE = 42

DeconvDesignMatrix = namedtuple("DeconvDesignMatrix",
                                ["predictors", "full", "n_lags", "rank", "singular_vals", "condition_num"])

def build_design(*, soa: float, sfreq: float,
                 n_words: int, n_trials: int,
                 response_lag: float,
                 rank_tol: float=1e-3,
                 intercept: bool=True,
                 verbose: bool=False,
                 **predictors):
    """note that the condition number is the design matrix X's condition number,
    so the model's normal-equation matrix condition number would be squared, i.e.,
    k(X^T X) = k(X)^2"""
    if predictors:
        for p in predictors.values():
            assert p.shape == (n_trials, n_words), "Each predictor must be a numpy array of shape (n_trials, n_words)"
    
    predictors = dict(predictors)
    if intercept:
        predictors = {"intercept": np.ones((n_trials, n_words), dtype=float), **predictors}
    
    ### TODO: need to check math.ceil() vs math.floor() vs int(round())
    # might need to check how decimation is done in mne.Epochs(decim), e.g., end points kept?
    n_time_samples_per_trial = math.ceil(soa * n_words * sfreq)
    n_time_samples = int(n_time_samples_per_trial * n_trials) 
    n_lags = int(round(response_lag * sfreq))
    soa_samples = math.ceil(soa * sfreq)
    if verbose:
        print(f"# time samples per trial = {n_time_samples_per_trial}")
        print(f"response to each sampled time point lasts {n_lags} time points")
    
    matrices = {name: np.zeros((n_time_samples, n_lags), dtype=float) for name in predictors}
    
    positions = range(n_words)
    for i in range(n_trials):
        for j in positions:
            valid_range = min(n_time_samples_per_trial - j * soa_samples, n_lags)
            row = i * n_time_samples_per_trial + j * soa_samples
            for m, p in zip(matrices.values(), predictors.values()):
                m[row : row + valid_range, : valid_range] += p[i, j] * np.eye(valid_range)

    X_full = np.hstack([m for m in matrices.values()])
    rank = np.linalg.matrix_rank(X_full, tol=rank_tol)
    singular_vals = np.linalg.svd(X_full, compute_uv=False)
    condition_num = np.linalg.cond(X_full)

    return DeconvDesignMatrix(matrices, X_full, n_lags, rank, singular_vals, condition_num)


class MySSP:
    # TODO: probably make it inherit from sklearn preprocessor or something
    # for better integration for future Amanda :D
    def __init__(self): 
        pass
    def fit_transform(self):
        pass
    def transform(self):
        pass


def refit(subj_data: np.ndarray, *,
          X_tune: np.ndarray, X_refit: np.ndarray,
          ch_roi: list,
          alpha_range: np.ndarray,
          lambda_common: float|None=None,
          score_on_mean: bool=True,
          ssp: bool=False, info: mne.Info|None=None,
          n_splits: int=5, ridge_tol: float=1e-4, solver: str="auto", max_iter: int|None=None,
          subj_data_mean: np.ndarray | None=None):
    """X_tune: n_rows = n_trials * n_time_samples_per_trial (vertically-stacked repeated block designed)
       X_refit: n_rows = n_time_samples_per_trial (one block of X_tune)"""
    # in standard OLS, fitting a vertically-stacked repeated block designed is equivalent to fitting one block of the design
    # to the block mean of the response vector
    # in Ridge, the equivalnece holds when alpha is divided by the number of blocks
    # (if not divided, the block-mean version will get over-regularized due to alpha being too large)

    RefitResults = namedtuple("RefitResults", ["raw_best_alpha", "refit_alpha",
                                               "coef", "pred", "r2", "solver", "n_iter"])
    
    if subj_data_mean is None:
        subj_data_mean = np.mean(subj_data, axis=0)

    if lambda_common is None:
        cv_results = tune_alpha(subj_data, X=X_tune,
                            alpha_range=alpha_range,
                            ssp=ssp, info=info,
                            n_splits=n_splits,
                            ridge_tol=ridge_tol,
                            solver=solver,
                            max_iter=max_iter,
                            score_on_mean=score_on_mean)
        raw_best_alpha = cv_results.best_alpha
        if score_on_mean:
            refit_alpha = raw_best_alpha
        else:
            refit_alpha = cv_results.best_alpha/subj_data.shape[0] # scale best alpha by trial count
    else:
        refit_alpha = lambda_common
        cv_results = None
        raw_best_alpha = None
    ridge = Ridge(alpha=refit_alpha, 
                  fit_intercept=False,         
                  tol=ridge_tol,
                  solver=solver,
                  positive=False,              
                  max_iter=max_iter,
                  random_state=RANDOM_STATE)
    
    n_roi_channels, n_time_samples = subj_data_mean.shape
    assert n_roi_channels == len(ch_roi)

    reconstruction = np.empty((n_roi_channels, n_time_samples))
    coefs = np.empty((n_roi_channels, X_refit.shape[1]))
    scores, solvers, n_iters = {}, {}, {}

    for i, ch in enumerate(ch_roi):
        ridge.fit(X_refit, subj_data_mean[i, :])
        reconstruction[i, :] = ridge.predict(X_refit)
        coefs[i, :] = ridge.coef_
        scores[ch] = ridge.score(X_refit, subj_data_mean[i, :])
        solvers[ch] = ridge.solver_
        n_iters[ch] = ridge.n_iter_

    refit_results = RefitResults(raw_best_alpha, refit_alpha,
                                 coefs, reconstruction, scores, solvers, n_iters)
    return cv_results, refit_results
    

def tune_alpha(ridge_data: np.ndarray, *, X: np.ndarray,  
               alpha_range: np.ndarray,
               ssp: bool=False, info: mne.Info | None=None,
               roi_idx: list | None=None,
               loo: bool=False, n_splits: int | None=None,
               ridge_tol: float=1e-4, solver: str="auto", max_iter: int|None=None,
               score_on_mean: bool=True,
               verbose: bool=False):
    """ridge_data.shape = (n_trials, n_channels, n_time_samples); n_time_samples is per trial
                          if doing LOO (e.g., training on 19 subjects, testing on 1), n_trials = n_subjects (but this does not work well)
       design matrix X.shape = (n_time_samples, n_predictors * n_lags)
       alpha_range in single-subject-level range (not divided by n_subj-1)
       if ssp is True, ridge_data should contain the full channel set needed for SSP, then only later restrict reporting / inference
       to ROI channels, where roi_idx must refer to the channel indices of the original full channel set.
       (If ridge_data contains only the ROI channels, then SSP is not very meaningful.)
       score_on_mean=True should NOT be used for a trial-specific permutation design where each trial has a different X_i, unless you explicitly intend to fit the averaged design matrix
    """
    
    ScoreResults = namedtuple("ScoreResults", ["train_scores", "train_mean", "train_std",
                                               "valid_scores", "valid_mean", "valid_std"])
    CVResults = namedtuple("CVResults", ["fold_scores", "best_mean_score", "best_alpha"])   
    
    n_trials, n_channels, n_time_samples_per_trial = ridge_data.shape
    assert n_trials * n_time_samples_per_trial  == X.shape[0]
    X_split = np.stack(np.split(X, n_trials), axis=0)    # stack into an array so advance indexing works for train_idx and valid_idx
                                                         # X_split.shape (n_trials, n_time_samples_per_trial, n_lags)
    if roi_idx is None:
        if verbose:
            print(f"ROI channel indices not set. Defaulting to all channels in the input data array (which might've been pre-selected via `epochs.get_data(picks=...)`).")
        roi_idx = list(range(n_channels))

    if loo:
        splitter = LeaveOneOut()
        n_splits = n_trials
    else:
        if n_splits is None:
            n_splits = 5
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    
    train_scores = np.empty((n_splits, len(alpha_range), len(roi_idx)))
    valid_scores = np.empty((n_splits, len(alpha_range), len(roi_idx)))

    for i, (train_idx, valid_idx) in enumerate(splitter.split(ridge_data)):
        if verbose:
            print(f"  FOLD {i+1}")
         
        X_train_stack = np.vstack(X_split[train_idx])
        X_valid_stack = np.vstack(X_split[valid_idx])

        if score_on_mean:
            X_train_mean = np.mean(X_split[train_idx], axis=0)
            X_valid_mean = np.mean(X_split[valid_idx], axis=0)

        y_train, y_valid = ridge_data[train_idx], ridge_data[valid_idx]  # y_train.shape = (n_train_trials, n_channels, n_time_samples_per_trial)  
        y_train_stack = np.hstack([y for y in y_train])    # shape = (n_channels, n_train_trials * n_time_samples_per_trial)
        y_valid_stack = np.hstack([y for y in y_valid])    # shape = (n_channels, n_valid_trials * n_time_samples_per_trial)

        if ssp:
            my_ssp = MySSP(info=info)
            y_train_stack = my_ssp.fit_transform(y_train_stack)       
            y_valid_stack = my_ssp.transform(y_valid_stack)       

            y_train = np.stack(np.split(y_train_stack, len(train_idx), axis=1), axis=0)
            y_valid = np.stack(np.split(y_valid_stack, len(valid_idx), axis=1), axis=0)
            # y_train = y_train_stack.reshape(y_train.shape)  <- not safe
            # y_valid = y_valid_stack.reshape(y_valid.shape)  <- not safe    
        
        for j, alpha in enumerate(alpha_range):
            if verbose:
                print(f"ALPHA = {alpha}")   

            ridge = Ridge(alpha=alpha,
                          fit_intercept=False,         
                          tol=ridge_tol,
                          solver=solver,
                          positive=False,              
                          max_iter=max_iter,
                          random_state=RANDOM_STATE)
            
            for k, ch in enumerate(roi_idx):
                if verbose:
                    print(f"    Channel Index {ch}")

                #ridge.fit(X_train_stack, y_train_stack[ch, :])

                if score_on_mean:
                    #X_train_mean = np.mean(X_split[train_idx], axis=0)  <- move outside of channel loop
                    #X_valid_mean = np.mean(X_split[valid_idx], axis=0)  <- move outside of channel loop
                    y_train_per_ch_mean = np.mean(y_train[:, ch, :], axis=0)
                    y_valid_per_ch_mean = np.mean(y_valid[:, ch, :], axis=0)
                    ridge.fit(X_train_mean, y_train_per_ch_mean)
                    train_scores[i, j, k] = ridge.score(X_train_mean,
                                                        y_train_per_ch_mean)
                    valid_scores[i, j, k] = ridge.score(X_valid_mean,
                                                        y_valid_per_ch_mean)
                else:
                    ridge.fit(X_train_stack, y_train_stack[ch, :])
                    train_scores[i, j, k] = ridge.score(X_train_stack,
                                                        y_train_stack[ch, :])
                    valid_scores[i, j, k] = ridge.score(X_valid_stack,
                                                        y_valid_stack[ch, :])
    
    fold_scores = {alpha: ScoreResults(train_scores[:, j, :], np.mean(train_scores[:, j, :]), np.std(train_scores[:, j, :]),
                                       valid_scores[:, j, :], np.mean(valid_scores[:, j, :]), np.std(valid_scores[:, j, :]))
                          for j, alpha in enumerate(alpha_range)}
            
    mean_scores = np.array([fold_scores[alpha].valid_mean for alpha in alpha_range])
    best_mean_score = np.max(mean_scores)
    best_alpha = alpha_range[np.argmax(mean_scores)]

    return CVResults(fold_scores, best_mean_score, best_alpha)


def nested_cv(subj_data: np.ndarray, *, X: np.ndarray,
              alpha_range: np.ndarray,
              ssp: bool=False, info: mne.Info | None=None,
              roi_idx: list | None=None,
              n_splits_outer: int=5, n_splits_inner: int=5,
              ridge_tol: float=1e-4, solver: str="auto", max_iter: int|None=None,
              score_on_mean: bool=True,
              verbose: bool=False):
    
    n_trials, n_channels, n_time_samples_per_trial = subj_data.shape
    assert n_trials * n_time_samples_per_trial  == X.shape[0]
    X_split_outer = np.stack(np.split(X, n_trials), axis=0)

    if roi_idx is None:
        if verbose:
            print(f"ROI channel indices not set. Defaulting to all channels in the input data array (which might've been pre-selected via `epochs.get_data(picks=...)`).")
        roi_idx = list(range(n_channels))

    kf = KFold(n_splits=n_splits_outer, shuffle=True, random_state=RANDOM_STATE)

    outer_scores = np.empty((n_splits_outer, len(roi_idx)))
    best_alphas = np.empty((n_splits_outer, ))
    cv_results_per_outer = np.empty((n_splits_outer, ), dtype=object)
    for i, (train_outer_idx, test_outer_idx) in enumerate(kf.split(subj_data)):
        if verbose:
            print(f"OUTER FOLD {i+1}")

        X_train_outer_stack = np.vstack(X_split_outer[train_outer_idx])
        X_test_outer_stack = np.vstack(X_split_outer[test_outer_idx])
        if score_on_mean:
            X_test_outer_mean = np.mean(X_split_outer[test_outer_idx], axis=0)
        
        y_train_outer = subj_data[train_outer_idx]
        y_test_outer = subj_data[test_outer_idx]

        cv_results = tune_alpha(y_train_outer, X=X_train_outer_stack,
                                alpha_range=alpha_range,
                                ssp=ssp, info=info,
                                roi_idx=roi_idx,
                                loo=False,
                                n_splits=n_splits_inner,
                                ridge_tol=ridge_tol,
                                score_on_mean=score_on_mean,
                                max_iter=max_iter,
                                solver=solver,
                                verbose=False)
        cv_results_per_outer[i] = cv_results
        best_alphas[i] = cv_results.best_alpha

        ridge = Ridge(alpha=cv_results.best_alpha,
                      fit_intercept=False,         
                      tol=ridge_tol,
                      solver=solver,
                      positive=False,              
                      max_iter=max_iter,
                      random_state=RANDOM_STATE)
        
        y_train_outer_stack = np.hstack([y for y in y_train_outer])
        y_test_outer_stack = np.hstack([y for y in y_test_outer])

        if ssp:
            my_ssp = MySSP(info=info)
            y_train_outer_stack = my_ssp.fit_transform(y_train_outer_stack)
            y_test_outer_stack = my_ssp.transform(y_test_outer_stack)
            # y_train_outer = np.stack(np.split(y_train_outer_stack, len(train_outer_idx), axis=1), axis=0) <- not actually used, leaving it here for debugging
            y_test_outer = np.stack(np.split(y_test_outer_stack, len(test_outer_idx), axis=1), axis=0)

        for j, ch in enumerate(roi_idx):
            ridge.fit(X_train_outer_stack, y_train_outer_stack[ch, :])

            if score_on_mean:
                # X_test_outer_mean = np.mean(X_split_outer[test_outer_idx], axis=0)  <- move outside channel loop
                y_test_outer_per_ch_mean = np.mean(y_test_outer[:, ch, :], axis=0)
                outer_scores[i, j] = ridge.score(X_test_outer_mean,
                                                 y_test_outer_per_ch_mean)
            else:
                outer_scores[i, j] = ridge.score(X_test_outer_stack,
                                                 y_test_outer_stack[ch, :])

    return outer_scores, best_alphas, cv_results_per_outer


def loop_over_subjs(*, epochs_iterable: list|np.ndarray,
                    matrices_full_iterable: list|np.ndarray,
                    matrices_restricted_iterable: list|np.ndarray|None,
                    ch_roi: list,
                    alpha_range: np.array,
                    intercept_name: str="intercept",
                    ssp: bool=False, info: mne.Info | None=None,
                    n_splits_outer: int=5, n_splits_inner: int=5,
                    ridge_tol: float=1e-4, solver: str="auto", max_iter: int|None=None,
                    verbose_subj_idx: bool=True,
                    verbose_tuning: bool=False) -> tuple:
    """Loops over all subjects to compare the full model with a restricted model."""
    
    outer_scores_full_per_subj = np.empty((len(epochs_iterable), ), dtype=object)
    outer_scores_restricted_per_subj = np.empty((len(epochs_iterable), ), dtype=object)
    cv_results_full_per_subj = np.empty((len(epochs_iterable), ), dtype=object)
    cv_results_restricted_per_subj = np.empty((len(epochs_iterable), ), dtype=object)
    best_alphas_per_subj = []
    
    if matrices_restricted_iterable is None:
        _matrices_restriced_iterable = np.empty(len(matrices_full_iterable),)
    
    for i, (epochs, matrix, _matrix) in enumerate(zip(epochs_iterable, matrices_full_iterable, _matrices_restriced_iterable)):
        if matrices_restricted_iterable is None:
            X_restricted = matrix.predictors[intercept_name]
        else:
            X_restricted = _matrix.full
        
        X_full = matrix.full
        subj_data = epochs.get_data(units="uV", picks=ch_roi)

        if verbose_subj_idx:
            print(f"subject index [{i}]...")
        if verbose_tuning:
            print(">> restricted")
        outer_scores_restricted, best_alphas_restricted, cv_results_restricted = nested_cv(subj_data, X=X_restricted,
                                                                  ssp=ssp, info=info,
                                                                  alpha_range=alpha_range,
                                                                  n_splits_outer=n_splits_outer, n_splits_inner=n_splits_inner,
                                                                  ridge_tol=ridge_tol, solver=solver, max_iter=max_iter,
                                                                  score_on_mean=False,
                                                                  verbose=verbose_tuning)
        outer_scores_restricted_per_subj[i] = outer_scores_restricted
        cv_results_restricted_per_subj[i] = cv_results_restricted
        if verbose_tuning:
            print(">> full model")
        outer_scores_full, best_alphas_full, cv_results_full = nested_cv(subj_data, X=X_full,
                                                       ssp=ssp, info=info,
                                                       alpha_range=alpha_range,
                                                       n_splits_outer=n_splits_outer, n_splits_inner=n_splits_inner,
                                                       ridge_tol=ridge_tol, solver=solver, max_iter=max_iter,
                                                       score_on_mean=False,
                                                       verbose=verbose_tuning)
        outer_scores_full_per_subj[i] = outer_scores_full
        cv_results_full_per_subj[i] = cv_results_full
        best_alphas_per_subj.append((best_alphas_restricted, best_alphas_full))
    
    LoopResults = namedtuple("LoopResults", ["outer_scores_full_per_subj",
                                             "outer_scores_restricted_per_subj",
                                             "cv_results_full_per_subj",
                                             "cv_results_restricted_per_subj",
                                             "best_alphas_per_subj"])
    loop_results = LoopResults(outer_scores_full_per_subj, outer_scores_restricted_per_subj,
                               cv_results_full_per_subj, cv_results_restricted_per_subj,
                               best_alphas_per_subj)
    return loop_results


def refit_all_subjs(*, epochs_iterable: list|np.ndarray,
                    evokeds_iterable: list|np.ndarray,
                    matrices_iterable: list|np.ndarray,
                    matrix_refit: DeconvDesignMatrix,
                    ch_roi: list,
                    alpha_range: np.ndarray|None,
                    lambda_common: float|None=None,
                    score_on_mean: bool=True,
                    ssp: bool=False, info: mne.Info|None=None,
                    n_splits: int=5,
                    ridge_tol: float=1e-4, solver: str="auto", max_iter: int|None=None,
                    verbose: bool=True) -> tuple:
    X_refit = matrix_refit.full
    if verbose:
        print(X_refit.shape)
    
    subj_tuned, subj_refit = [], []
    for i, (epochs, evoked, matrix) in enumerate(zip(epochs_iterable, evokeds_iterable, matrices_iterable)):
        if verbose:
            print(f"subject index [{i}]...")
        subj_epochs_data = epochs.get_data(units="uV", picks=ch_roi)
        subj_evoked_data = evoked.get_data(units="uV", picks=ch_roi)
        X_tune = matrix.full
        cv_results, refit_results = refit(subj_epochs_data,
                                          X_tune=X_tune, X_refit=X_refit,
                                          ch_roi=ch_roi,
                                          alpha_range=alpha_range,
                                          lambda_common=lambda_common,
                                          score_on_mean=score_on_mean,
                                          ssp=ssp, info=info,
                                          n_splits=n_splits,
                                          ridge_tol=ridge_tol,
                                          solver=solver,
                                          max_iter=max_iter,
                                          subj_data_mean=subj_evoked_data)
        subj_tuned.append(cv_results)
        subj_refit.append(refit_results)
    return subj_tuned, subj_refit


def compare_model_fit(outer_scores_full_per_subj: np.ndarray,
                      outer_scores_restricted_per_subj: np.ndarray,
                      ch_roi: list) -> pd.DataFrame:
    """outer_scores_full_per_subj.shape = outer_scores_restricted_per_subj.shape = (n_subj,)"""
    
    delta_r2_per_subj = outer_scores_full_per_subj - outer_scores_restricted_per_subj
    delta_r2 = np.array([d for d in delta_r2_per_subj])       # shape (n_subj, n_splits_outer, n_ch_roi)
    delta_r2_fold_mean = np.mean(delta_r2, axis=1)            # shape (n_subj, n_ch_roi)

    outer_scores_full = np.array([s for s in outer_scores_full_per_subj])
    outer_scores_restricted = np.array([s for s in outer_scores_restricted_per_subj])
    outer_scores_full_mean = np.mean(np.mean(outer_scores_full, axis=1), axis=0)
    outer_scores_restricted_mean = np.mean(np.mean(outer_scores_restricted, axis=1), axis=0)

    improve = [ttest_1samp(delta_r2_fold_mean[:, i], 0.0, alternative="greater")
                                           for i in range(delta_r2.shape[-1])]
    ts = [im.statistic for im in improve]
    ps = [im.pvalue for im in improve]
    _, qs = fdrcorrection(ps, alpha=0.01, method="indep", is_sorted=False)
    improve_df = pd.DataFrame({"R^2 (full)": outer_scores_full_mean,
                               "R^2 (intercept)": outer_scores_restricted_mean,
                               "delta R^2": np.mean(delta_r2_fold_mean, axis=0),
                               "t-statistic": ts, "p-val": ps, "q-val (fdr)": qs})
    improve_df = improve_df.rename(index={ch_roi.index(ch): ch for ch in ch_roi})
    return improve_df


def get_time_window_coefs(subj_refit: list, *,
                          n_predictors: int,
                          intercept_index: int|None,
                          beta_index: int,
                          time_window: tuple,  # time_window in seconds, e.g., for N400, (0.3, 0.6)
                          sfreq: float,
                          sig_thresh: float=0.05): 
    """subj_refit is a list of RefitResults (returned by function `refit()`) obtained from looping over subjects"""
    WindowCoefs = namedtuple("WindowCoefs", ["beta", "intercept",
                                             "beta_window", "intercept_window",
                                             "beta_ts", "beta_ps", "beta_qs", "reject", "sig_thresh"])
    
    coefs = np.array([subj_refit[i].coef for i in range(len(subj_refit))])

    coefs_split = np.split(coefs, n_predictors, axis=-1)
    beta = coefs_split[beta_index]
    
    # TODO: handle beta indexing so it still works if pre word1 onset segments are also modeled
    window = int(round(time_window[0] * sfreq)), int(round(time_window[1] * sfreq))
    beta_window = beta[:, :, window[0]: window[1]+1]
    if intercept_index is not None:
        intercept = coefs_split[intercept_index]
        intercept_window = intercept[:, :, window[0]: window[1]+1]
    else:
        intercept, intercept_window = None, None

    beta_ts, beta_ps = ttest_1samp(beta_window, popmean=0, axis=0)
    reject, beta_qs = fdrcorrection(beta_ps.ravel(), alpha=sig_thresh)
    reject = reject.reshape(beta_ps.shape)
    beta_qs = beta_qs.reshape(beta_ps.shape)

    window_coefs = WindowCoefs(beta, intercept,
                               beta_window, intercept_window,
                               beta_ts, beta_ps, beta_qs, reject, sig_thresh)
    return window_coefs


def time_window_heatmap(window_coefs, *,
                        time_window: tuple,
                        dt: int,
                        ch_roi: list,
                        show_corrected: bool=True,
                        figsize: tuple=(7, 3),
                        cmap: str="RdBu_r",
                        vmin: float|None=None,
                        vmax: float|None=None,
                        s: float=20.0,
                        title_fontsize: float=10.0,
                        xy_label_fontsize: float=8.0,
                        tick_labelsize: float=7.5,
                        pad: float=0.02,
                        cbar_label_fontsize: float=7.0,
                        component_name: str="N400",
                        emotion_category: str="All Emotions Collapsed",
                        alt_title: str|None=None):
    
    beta_window = window_coefs.beta_window
    times_ms = np.arange(int(time_window[0]*1000), int(time_window[1]*1000)+1, dt)
    assert beta_window.shape[2] == len(times_ms)

    heat_data = beta_window.mean(axis=0)  # shape: (n_ch_roi, n_time_window_time_points) subject-mean beta heatmap: channels × time
    if show_corrected:
        sig_mask = window_coefs.reject
        cor = "FDR-Corrected"
    else:
        sig_mask = window_coefs.beta_ps < window_coefs.sig_thresh
        cor = f"Non-Corrected (p < {window_coefs.sig_thresh})"

    fig, axes = plt.subplots(figsize=figsize)

    # pcolormesh wants bin edges, not centers
    time_edges = np.r_[times_ms - dt / 2, times_ms[-1] + dt / 2]
    channel_edges = np.arange(len(ch_roi) + 1)
    if vmax is None and vmin is None:
        vmax = np.nanmax(np.abs(heat_data))
    mesh = axes.pcolormesh(time_edges,
                           channel_edges,
                           heat_data,
                           shading="auto",
                           cmap=cmap,
                           vmin=-vmax,
                           vmax=vmax)
    yy, xx = np.where(sig_mask)
    axes.scatter(times_ms[xx], yy+0.5, marker=".", s=s, color="black", linewidth=0)

    axes.set_yticks(np.arange(len(ch_roi)) + 0.5)
    axes.set_yticklabels(ch_roi)
    axes.invert_yaxis()

    axes.set_xticks(np.arange(int(time_window[0]*1000), int(time_window[1]*1000)+1, 5*dt))
    axes.set_xlabel("Lag from word onset (ms)", fontsize=xy_label_fontsize)
    axes.set_ylabel("Channel", fontsize=xy_label_fontsize)
    if alt_title is None:
        axes.set_title(f"Mean Subject-Wise CV-Tuned Ridge Coefficient for Log Word Position in {component_name} Window,\n{emotion_category}. Black Dots Indicate Significance, {cor}", fontsize=title_fontsize)
    else:
        axes.set_title(alt_title, fontsize=title_fontsize)

    axes.set_xticks(times_ms, minor=True)
    axes.set_yticks(np.arange(len(ch_roi)) + 1, minor=True)
    # axes.grid(which="minor", color="white", linewidth=0.45)
    axes.tick_params(axis="both", which="minor", bottom=False, left=False, labelsize=tick_labelsize)

    cbar = fig.colorbar(mesh, ax=axes, pad=pad)
    cbar.set_label("Beta coefficient", fontsize=cbar_label_fontsize)

    fig.tight_layout()
    return fig

