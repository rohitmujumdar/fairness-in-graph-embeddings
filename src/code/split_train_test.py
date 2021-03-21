import pandas as pd
from sklearn.model_selection import train_test_split

df_positive = pd.read_csv('../data/df_positive.csv', index_col=0)
df_negative = pd.read_csv('../data/df_negative.csv', index_col=0)

df_positive_train, df_positive_test = train_test_split(df_positive,
                                                       train_size=0.8,
                                                       stratify=df_positive[
                                                           ['userId']],
                                                       random_state=0)
assert df_positive_train.isnull().values.any() == False
assert df_positive_test.isnull().values.any() == False

df_negative_train, df_negative_test = train_test_split(df_negative,
                                                       train_size=len(
                                                           df_positive_train),
                                                       stratify=df_negative[
                                                           ['userId']],
                                                       random_state=0)

assert df_negative_train.isnull().values.any() == False
assert df_negative_train.isnull().values.any() == False

assert len(df_negative_train) == len(df_positive_train)

df_train = pd.concat([df_positive_train, df_negative_train])
df_test = pd.concat([df_positive_test, df_negative_test])

assert df_train.isnull().values.any() == False
assert df_test.isnull().values.any() == False

df_train.to_csv('../data/df_train.csv')
df_test.to_csv('../data/df_test.csv')
