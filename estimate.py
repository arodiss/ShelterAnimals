import pandas as pd
from sklearn.metrics import log_loss


def estimate(alias, predictions_file):
    print("Estimating " + alias + '...')
    real_cv = pd.read_csv('normalized/cross_validation.csv')
    predicted_cv = pd.read_csv(predictions_file)

    real_cv['Transfer'] = real_cv['outcome'] == 'Transfer'
    real_cv['Return_to_owner'] = real_cv['outcome'] == 'Return_to_owner'
    real_cv['Euthanasia'] = real_cv['outcome'] == 'Euthanasia'
    real_cv['Adoption'] = real_cv['outcome'] == 'Adoption'
    real_cv['Died'] = real_cv['outcome'] == 'Died'

    print('MLogLoss: ' + str(log_loss(
        real_cv.loc[:, ('Transfer', 'Return_to_owner', 'Euthanasia', 'Adoption', 'Died')].astype(float).as_matrix(),
        predicted_cv.loc[:, ('Transfer', 'Return_to_owner', 'Euthanasia', 'Adoption', 'Died')].astype(float).as_matrix()
    )))

estimate('xgboost', 'predictions_cv/xgboost.csv')
estimate('nn', 'predictions_cv/nn.csv')
estimate('svm', 'predictions_cv/svm.csv')
estimate('ensemble', 'predictions_cv/ensemble.csv')
