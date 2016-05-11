Shelter Animals Outcome
====

This is a script for Kaggle competition [Shelter Animals Outcome](https://www.kaggle.com/c/shelter-animal-outcomes)

### How to run locally

Flow is simple - normalize data (i.e. mine features), run a set of prediction algos, ensemble their outcomes, estimate the result
```
python3 normalize.py
python3 classify-xgboost.py
python3 classify-nn.py
python3 classify-svm.py
python3 ensemble.py
python3 estimate.py
```
