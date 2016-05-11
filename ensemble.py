import pandas as pd


weight_svm = 0.06
weight_nn = 0.18
weight_xgboost = 1 - weight_nn - weight_svm


def ensemble(xgboost_file, nn_file, svm_file, output_file):
    result_xgboost = pd.read_csv(xgboost_file)
    result_nn = pd.read_csv(nn_file)
    result_svm = pd.read_csv(svm_file)

    result = result_xgboost * weight_xgboost + result_nn * weight_nn + result_svm * weight_svm
    result.to_csv(output_file, float_format='%.5f')

print("Ensembling...")
ensemble('predictions_cv/xgboost.csv', 'predictions_cv/nn.csv', 'predictions_cv/svm.csv', 'predictions_cv/ensemble.csv')
ensemble('predictions/xgboost.csv', 'predictions/nn.csv', 'predictions/svm.csv', 'predictions/ensemble.csv')
print("Ensembled")