import numpy as np
import pandas as pd
import skmultiflow as sm
import itertools

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.neural_network import MLPClassifier

from xEvaluator import XEvaluator

# CONFIGURATION ---------
SCALE = True
RANDOM_STATE=500
PRET_SIZE=100
BASELINE_WIN=100
RESULTS_PATH=(Path.cwd() / 'results').resolve()
# -----------------------

def run_experiments(path, model_names, datasets, explainers, pret_size, baseline_win):
            df = pd.DataFrame(columns=['Classifier', 'Data set', 'Accuracy', 'Explainer', 'Stability', 
                                       'Attribution Discrepancy', 'Top-10 Presence', 'Top-10 Similarity', 'Time'])

            get_nsamples = lambda num_features, sample_size: sample_size*num_features + 1 if sample_size is not None else None

            for model in model_names:
                for ds in datasets:
                    for expl in explainers:
                        nsamples = get_nsamples(datasets[ds][0].n_features, expl[3])

                        if model == 'Logistic Regression':
                            if expl[0] == 'custom':
                                evaluator = XEvaluator(datasets[ds][0], SGDClassifier(random_state=RANDOM_STATE, loss='log'), expl[0], 
                                                   baseline_window=baseline_win, pretrain_size=pret_size, max_batches=datasets[ds][1], 
                                                   param_samples=expl[2], results_dir=(path / (model+ds)).resolve(), ensemble=expl[1], stream_name=ds, use_proba=False,
                                                   custom_data={'priors': [np.zeros((datasets[ds][0].n_features+1, 1)), np.identity(datasets[ds][0].n_features+1), None]},
                                                   custom_baseline=True, clf_name=model, plot=False, expl_nsamples=nsamples)
                            else:
                                evaluator = XEvaluator(datasets[ds][0], SGDClassifier(random_state=RANDOM_STATE, loss='log'), expl[0], 
                                                       baseline_window=baseline_win, pretrain_size=pret_size, max_batches=datasets[ds][1], 
                                                       param_samples=expl[2], results_dir=(path / (model+ds)).resolve(), ensemble=expl[1], stream_name=ds, use_proba=False,
                                                       clf_name=model, expl_nsamples=nsamples)
                        elif model == 'Perceptron':
                            if expl[0] == 'custom':
                                evaluator = XEvaluator(datasets[ds][0], Perceptron(random_state=RANDOM_STATE), expl[0], 
                                                   baseline_window=baseline_win, pretrain_size=pret_size, max_batches=datasets[ds][1], 
                                                   param_samples=expl[2], results_dir=(path / (model+ds)).resolve(), ensemble=expl[1], stream_name=ds, use_proba=False,
                                                   custom_data={'priors': [np.zeros((datasets[ds][0].n_features+1, 1)), np.identity(datasets[ds][0].n_features+1), None]},
                                                   custom_baseline=True, clf_name=model, plot=False, expl_nsamples=nsamples)
                            else:
                                evaluator = XEvaluator(datasets[ds][0], Perceptron(random_state=RANDOM_STATE), expl[0], 
                                                       baseline_window=baseline_win, pretrain_size=pret_size, max_batches=datasets[ds][1], 
                                                       param_samples=expl[2], results_dir=(path / (model+ds)).resolve(), ensemble=expl[1], stream_name=ds, use_proba=False,
                                                       clf_name=model, expl_nsamples=nsamples)
                        elif model =='MLPClassifier':
                            if expl[0] == 'custom':
                                evaluator = XEvaluator(datasets[ds][0], MLPClassifier([32, 16], random_state=RANDOM_STATE), expl[0], 
                                                   baseline_window=baseline_win, pretrain_size=pret_size, max_batches=datasets[ds][1], 
                                                   param_samples=expl[2], results_dir=(path / (model+ds)).resolve(), ensemble=expl[1], stream_name=ds, use_proba=False,
                                                   custom_data={'priors': [np.zeros((datasets[ds][0].n_features+1, 1)), np.identity(datasets[ds][0].n_features+1), None]},
                                                   custom_baseline=True, clf_name=model, plot=False, ann=True, expl_nsamples=nsamples)
                            else:
                                evaluator = XEvaluator(datasets[ds][0], MLPClassifier([32, 16], random_state=RANDOM_STATE), expl[0], 
                                                       baseline_window=baseline_win, pretrain_size=pret_size, max_batches=datasets[ds][1], 
                                                       param_samples=expl[2], results_dir=(path / (model+ds)).resolve(), ensemble=expl[1], stream_name=ds, use_proba=False,
                                                       clf_name=model, ann=True, plot=False, expl_nsamples=nsamples)

                        df.loc[len(df)] = evaluator.evaluate()
                        print(df)

                        df.to_csv((path / 'experiment_results.csv').resolve())
                        df.to_latex((path / 'experiment_results.tex').resolve(), index=False,  float_format='{:,.2f}'.format)


def prepare_streams(path, scale=SCALE):
    RBF_stream = sm.data.RandomRBFGenerator(model_random_state=RANDOM_STATE,
                                            sample_random_state=RANDOM_STATE,
                                            n_classes=2,
                                            n_features=100)

    RTG_stream = sm.data.RandomTreeGenerator(tree_random_state=RANDOM_STATE,
                                             sample_random_state=RANDOM_STATE,
                                             n_classes=2,
                                             n_cat_features=10,
                                             n_num_features=50,
                                             n_categories_per_cat_feature=5)


    spambase = pd.read_csv('datasets/spambase.csv').sample(frac=1, random_state=RANDOM_STATE)
    spam_pos = spambase.iloc[:,-1].value_counts()[1]

    card_default = pd.read_csv('datasets/default_of_credit_card_clients.csv', header=1, index_col=0)
    card_default_pos = card_default.iloc[:5001,-1].value_counts()[1]


    X_rbf, y_rbf = RBF_stream.next_sample(10_000)
    RBF_pos = np.unique(y_rbf[:5001], return_counts=True)[1][1]

    X_rtg, y_rtg = RTG_stream.next_sample(10_000)
    RTG_pos = np.unique(y_rtg[:5001], return_counts=True)[1][1]

    if scale:
        spambase[spambase.columns[:-1]] = MinMaxScaler().fit_transform(spambase[spambase.columns[:-1]])
        card_default[card_default.columns[:-1]] = MinMaxScaler().fit_transform(card_default[card_default.columns[:-1]])

        RBF = np.column_stack((MinMaxScaler().fit_transform(X_rbf), y_rbf))
        RTG = np.column_stack((MinMaxScaler().fit_transform(X_rtg), y_rtg))
    else:
        RBF = np.column_stack((X_rbf, y_rbf))
        RTG = np.column_stack((X_rtg, y_rtg))


    df_RBF = pd.DataFrame(RBF, columns=[f'feature {d}' for d in range(X_rbf.shape[1])]+['y']).astype({'y': int})
    df_RTG = pd.DataFrame(RTG, columns=[f'feature {d}' for d in range(X_rtg.shape[1])]+['y']).astype({'y': int})

    RTG_stream = sm.data.DataStream(df_RTG)
    RBF_stream = sm.data.DataStream(df_RBF)
    spam_stream = sm.data.DataStream(spambase)
    card_default_stream = sm.data.DataStream(card_default)

    datasets_df = pd.DataFrame(columns=['Data set', 'Samples', 'Features', 'Positive Class'])
    datasets_df.loc[len(datasets_df)] = ['RBF', 5_000, RBF_stream.n_features, RBF_pos]
    datasets_df.loc[len(datasets_df)] = ['RTG', 5_000, RTG_stream.n_features, RTG_pos]
    datasets_df.loc[len(datasets_df)] = ['Spambase', 4_601, spam_stream.n_features, spam_pos]
    datasets_df.loc[len(datasets_df)] = ['Card Default', 5_000, card_default_stream.n_features, card_default_pos]

    path.mkdir(parents=True, exist_ok=True)
    datasets_df.to_csv((path / 'experiment_datasets.csv').resolve())
    datasets_df.to_latex((path / 'experiment_datasets.tex').resolve(), index=False)


    return RTG_stream, RBF_stream, spam_stream, card_default_stream


def ensemble_samples_evaluation(model_names, datasets, ensemble_sizes, sample_sizes, path=(RESULTS_PATH / 'results_sample_sizes').resolve()):
    path.mkdir(parents=True, exist_ok=True)

    explainers = []
    for e_size, s_size in itertools.product(ensemble_sizes, sample_sizes):
        for expl in ['shap', 'lime']:
            explainers.append((expl, e_size > 1, e_size, s_size))
    run_experiments(path, model_names, datasets, explainers, pret_size=PRET_SIZE, baseline_win=BASELINE_WIN)


if __name__ =='__main__':

    RTG_stream, RBF_stream, spam_stream, card_default_stream = prepare_streams(RESULTS_PATH, SCALE)

    # Generate paper results
    model_names = ['Logistic Regression', 'Perceptron', 'MLPClassifier']
    
    datasets = ['RTG', 'RBF', 'Spambase', 'Credit default']
    streams = [(RTG_stream, 5000), (RBF_stream, 5000), (spam_stream, 4600-PRET_SIZE), (card_default_stream, 5000)]
    datasets = dict(zip(datasets, streams))

    explainers = [('shap', False, 1, None), ('shap', True, 20, None), (['fires', 'shap'], False, 1, None), 
                  ('lime', False, 1, None), ('lime', True, 20, None), (['fires', 'lime'], False, 1, None), 
                  ('fires', False, 1, None), 
                  ('custom', False, 1, None)]

    RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    # Main results
    run_experiments(RESULTS_PATH, model_names, datasets, explainers, pret_size=PRET_SIZE, baseline_win=BASELINE_WIN)
    
    # Results for varying number of samples in local surrogate methods and ensembles of models
    ensemble_samples_evaluation(['MLPClassifier', 'Logistic Regression'], datasets, ensemble_sizes=[1, 10], sample_sizes=[2, 10])
