import pandas as pd
from lasagne import layers
from lasagne.nonlinearities import softmax, rectify, sigmoid
from nolearn.lasagne import NeuralNet

features = [
    'has_name',
    'age',
    'month', 'wday', 'hour', 'imonth', 'iwday', 'ihour',
    'is_intact_male', 'is_neutered_male', 'is_intact_female', 'is_spayed_female',
    'is_red', 'is_gray', 'is_tricolor', 'is_tabby', 'is_mixed_color',
    'is_cat', 'is_mix', 'is_pit_bull'
]


def classify(train_file, test_file, output_file):
    test = pd.read_csv(test_file)
    train = pd.read_csv(train_file)
    test['age'] = test['age'] / train['age'].max()
    train['age'] = train['age'] / train['age'].max()
    train.replace(to_replace={'Transfer': 0, 'Return_to_owner': 1, 'Euthanasia': 2, 'Adoption': 3, 'Died': 4}, inplace=True)
    test.replace(to_replace={'Transfer': 0, 'Return_to_owner': 1, 'Euthanasia': 2, 'Adoption': 3, 'Died': 4}, inplace=True)

    train_X = train.ix[:, features].as_matrix().astype('float32')
    train_Y = train.ix[:, 'outcome'].as_matrix().astype('int32')
    test_X = test.ix[:, features].as_matrix().astype('float32')

    model = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('hidden1', layers.DenseLayer),
            ('hidden2', layers.DenseLayer),
            ('output', layers.DenseLayer),
        ],

        input_shape=(None, len(features)),
        hidden1_num_units=100, hidden1_nonlinearity=sigmoid,
        hidden2_num_units=100, hidden2_nonlinearity=rectify,
        max_epochs=100,
        output_nonlinearity=softmax,
        output_num_units=5,
        update_learning_rate=0.01,
    ).fit(train_X, train_Y)
    all_proba = model.predict_proba(test_X).reshape(test_X.shape[0], 5)

    result = pd.DataFrame(
        data=all_proba,
        columns=['Transfer', 'Return_to_owner', 'Euthanasia', 'Adoption', 'Died'],
        index=test['id']
    )
    result.to_csv(output_file, index_label="ID", float_format='%.5f')


print('Classifying with NN...')
classify('normalized/train.csv', 'normalized/cross_validation.csv', 'predictions_cv/nn.csv')
classify('normalized/train_all.csv', 'normalized/test.csv', 'predictions/nn.csv')
print('Classified')