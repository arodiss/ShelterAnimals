import pandas as pd
import numpy as np
import datetime


#todo RF
def normalize_breed(df):
    df['is_cat'] = df['AnimalType'] == 'Cat'
    df['is_mix'] = 'Mix' in df['Breed'] or '/' in df['Breed']
    df['is_pit_bull'] = 'Pit Bull' in df['Breed']
    del df['AnimalType']
    del df['Breed']


def normalize_age_raw(raw_age):
    if raw_age is None or isinstance(raw_age, float):
        return 790 #average value
    else:
        [units_count, unit_name] = raw_age.split(' ')
        return {
            'year': lambda x: x * 365,
            'years': lambda x: x * 365,
            'month': lambda x: x * 30,
            'months': lambda x: x * 30,
            'week': lambda x: x * 7,
            'weeks': lambda x: x * 7,
            'day': lambda x: x,
            'days': lambda x: x,
        }[unit_name](int(units_count))


def normalize_age(df):
    df['age'] = df['AgeuponOutcome'].map(normalize_age_raw)
    del df['AgeuponOutcome']


def normalize_color(df):
    df['is_red'] = 'Red' in df['Color'] or 'red' in df['Color']
    df['is_gray'] = 'Gray' in df['Color'] or 'gray' in df['Color'] or 'Blue' in df['Color'] or 'blue' in df['Color']
    df['is_tricolor'] = 'Tricolor' in df['Color'] or 'tricolor' in df['Color']
    df['is_tabby'] = 'Tabby' in df['Color'] or 'tabby' in df['Color']
    df['is_mixed_color'] = 'Mix' in df['Color'] or 'mix' in df['Color'] or '/' in df['Color']
    del df['Color']


def normalize_sex(df):
    df['is_intact_male'] = df['SexuponOutcome'] == 'Intact Male'
    df['is_neutered_male'] = df['SexuponOutcome'] == 'Neutered Male'
    df['is_intact_female'] = df['SexuponOutcome'] == 'Intact Female'
    df['is_spayed_female'] = df['SexuponOutcome'] == 'Spayed Female'
    del(df['SexuponOutcome'])


def normalize_name(df):
    df['has_name'] = df['Name'].astype(bool)
    del(df['Name'])


def normalize_time(df):
    df['time'] = pd.to_datetime(df['DateTime'])
    del df['DateTime']
    df['hour'] = df['time'].map(lambda time_obj: time_obj.hour)
    df['wday'] = df['time'].map(lambda time_obj: time_obj.isoweekday())
    df['month'] = df['time'].map(lambda time_obj: time_obj.month)
    del df['time']
    df['ihour'] = df['hour'].map(lambda hour: hour - 12 if hour > 12 else 12 - hour)
    df['iwday'] = df['wday'].map(lambda wday: wday - 4 if wday > 4 else 4 - wday)
    df['imonth'] = df['month'].map(lambda month: month - 6 if month > 6 else 6 - month)


def normalize_df(df):
    if 'ID' in df:
        df = df.set_index('ID')
    else:
        df = df.set_index('AnimalID')
    normalize_sex(df)
    normalize_name(df)
    normalize_breed(df)
    normalize_time(df)
    normalize_color(df)
    normalize_age(df)
    df = df.reindex(df.index.rename('id'))
    if 'OutcomeType' in df:
        df['outcome'] = df['OutcomeType']
        del df['OutcomeType']
        del df['OutcomeSubtype']

    return df


print('Normalizing train dataset...')
df = pd.read_csv('data/train.csv')
normalized = normalize_df(df)
frame_mask = np.random.rand(len(normalized)) < 0.8
train = normalized[frame_mask]
cross_validation = normalized[~frame_mask]
normalized.to_csv('normalized/train_all.csv')
train.to_csv('normalized/train.csv')
cross_validation.to_csv('normalized/cross_validation.csv')

print('Normalizing test dataset...')
df = pd.read_csv('data/test.csv', sep=',')
# normalized = pd.DataFrame.from_records(df.apply(lambda row: normalize_row(row), axis=1))
normalized = normalize_df(df)
normalized.to_csv('normalized/test.csv')

print('Normalization complete')
