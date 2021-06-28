# XEvaluator

Environment for evaluation of explainability methods in data streams.

Results included in *Approaches to Robust Online Explanations* paper can be found in `results` directory.
They were generated with `experiments.py`.

NOTICE: Not all of the configurations of the evaluator work. There are still some missing checks.

## Features:
- Extensible
    - inclusion of additional attribution schemes
    - for example, an attribution scheme which assumes that output `y = ax` is linearly dependent on attributions (`a`) and data points (`x`) with Gaussian prior and likelihood
- Visualization
    - allows for visual inspection of results
- Real-time Explanations
    - keep track of global (cumulative) feature attributions, attribution stability and local quality in real-time
- Weighted Combination of Explainers
    - following explainers are built-in and can immediately be combined:
        - [SHAP](https://github.com/slundberg/shap)
        - [FIRES](https://github.com/haugjo/fires)
        - [LIME](https://github.com/marcotcr/lime)
    - custom attribution schemes can also be combined

### Limitations:
- only tabular data
- only classifiers with 2 classes

#### Visualization
Real-time visualization shows 4 subplots:

1. *Accuracy* -- top-left -- accuracy of the classifier in defined window of samples
2. *Stability and Local Explanation Quality* -- top-right -- cumulative stability of the explainer and local explanation quality
3. *Global Feature Attribution* -- middle -- graph with top-K features with the highest cumulative attributions assigned by the explainer sorted in descending order
4. *Ground Truth* -- bottom -- weights of the model considered to be the ground truth for feature attribution by the classifier. Features are sorted in descending order.

## Usage

### Dependencies
```
numpy
matplotlib
fires
lime
shap
```
For stream creation:
`skmultiflow`

To install dependencies call:
`pip install -r requirements.txt`
and install [fires](https://github.com/haugjo/fires).

### Example
```python
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
```