import skmultiflow as sm

from xEvaluator import XEvaluator
from sklearn.linear_model import SGDClassifier


# Create data streams

# Synthetic data stream:
stream = sm.data.RandomRBFGenerator(model_random_state=500,
                                            sample_random_state=500,
                                            n_classes=2,
                                            n_features=100)

# Or create a data stream from your data
# For example:
# stream = sm.data.DataStream(pd.read_csv('path/my_data.csv'))

# Example with logistic regression classifier and shap explainer
classifier = SGDClassifier(random_state=500, loss='log')
evaluator = XEvaluator(stream,
                       classifier, 
                       'shap', 
                       baseline_window=100, 
                       pretrain_size=100, 
                       max_batches=500, 
                       param_samples=1, 
                       results_dir='logReg_rbf')
evaluator.evaluate()