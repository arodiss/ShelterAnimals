import xgboost as xgb
import pandas as pd


features = [
    'age', 'has_name',
    'month', 'wday', 'hour', 'imonth', 'iwday', 'ihour',
    'is_intact_male', 'is_neutered_male', 'is_intact_female', 'is_spayed_female',
    'is_red', 'is_gray', 'is_tricolor', 'is_tabby', 'is_mixed_color',
    'is_cat', 'is_mix', 'is_pit_bull'
]


def classify(train_file, test_file, output_file):
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    train.replace(to_replace={'Transfer': 0, 'Return_to_owner': 1, 'Euthanasia': 2, 'Adoption': 3, 'Died': 4}, inplace=True)
    test.replace(to_replace={'Transfer': 0, 'Return_to_owner': 1, 'Euthanasia': 2, 'Adoption': 3, 'Died': 4}, inplace=True)

    train_X = train.ix[:, features]
    train_Y = train.ix[:, 'outcome']

    test_X = test.ix[:, features]

    xg_train = xgb.DMatrix(train_X, label=train_Y)
    xg_test = xgb.DMatrix(test_X)

    model = xgb.train(
        {
            'eval_metric': 'mlogloss',
            'eta': 0.25,
            'max_depth': 10,
            'silent': 1,
            'nthread': 4,
            'num_class': 5,
            'objective': 'multi:softprob'
        },
        xg_train,
        num_boost_round=25
    )
    all_proba = model.predict(xg_test).reshape(test_X.shape[0], 5)

    result = pd.DataFrame(
        data=all_proba,
        columns=['Transfer', 'Return_to_owner', 'Euthanasia', 'Adoption', 'Died'],
        index=test['id']
    )
    result.to_csv(output_file, index_label="ID", float_format='%.5f')


print('Classifying with XGBoost...')
classify('normalized/train.csv', 'normalized/cross_validation.csv', 'predictions_cv/xgboost.csv')
classify('normalized/train_all.csv', 'normalized/test.csv', 'predictions/xgboost.csv')
print('Classified')