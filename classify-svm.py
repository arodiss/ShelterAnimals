import pandas as pd
from sklearn.svm import SVC


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

    model = SVC(probability=True).fit(train_X, train_Y)
    all_proba = model.predict_proba(test_X).reshape(test_X.shape[0], 5)

    result = pd.DataFrame(
        data=all_proba,
        columns=['Transfer', 'Return_to_owner', 'Euthanasia', 'Adoption', 'Died'],
        index=test['id']
    )
    result.to_csv(output_file, index_label="ID", float_format='%.5f')


print('Classifying with SVM...')
classify('normalized/train.csv', 'normalized/cross_validation.csv', 'predictions_cv/svm.csv')
classify('normalized/train_all.csv', 'normalized/test.csv', 'predictions/svm.csv')
print('Classified')