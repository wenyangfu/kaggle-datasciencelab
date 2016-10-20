# Taken from
# https://www.kaggle.com/fchollet/flavours-of-physics/keras-starter-code-deep-pyramidal-mlp/code

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils

from sklearn.preprocessing import StandardScaler, Imputer


def impute_missing(X):
    # Impute missing features

    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    train_xform = imp.fit_transform(X)
    return train_xform, imp


def get_training_data():
    train = pd.read_csv('data/train_final.csv')
    ids = train.id
    train = train.drop(['id'], axis=1)

    # Divide dataset into X and y
    y = train.Y
    X = train.drop(["Y"], axis=1)

    return (ids, X.as_matrix(), y.as_matrix())


def get_test_data():
    test = pd.read_csv('data/test_final.csv')
    ids = test.id
    test = test.drop(['id'], axis=1)
    return ids.as_matrix(), test.as_matrix()


def preprocess_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler


# get training data
ids, X, y = get_training_data()
print('Data shape:', X.shape)

# shuffle the data
np.random.seed(1337)
np.random.shuffle(X)
np.random.seed(1337)
np.random.shuffle(y)

print('Signal ratio:', np.sum(y) / y.shape[0])

# preprocess the data
X, imputer = impute_missing(X)
X, scaler = preprocess_data(X)
y = np_utils.to_categorical(y)

# split into training / evaluation data
nb_train_sample = int(len(y) * 0.80)
X_train = X[:nb_train_sample]
X_eval = X[nb_train_sample:]
y_train = y[:nb_train_sample]
y_eval = y[nb_train_sample:]

print('Train on:', X_train.shape[0])
print('Eval on:', X_eval.shape[0])

# deep pyramidal MLP, narrowing with depth
model = Sequential()
model.add(Dropout(0.13, input_shape=(X_train.shape[1],)))
model.add(Dense(75))
model.add(PReLU())

model.add(Dropout(0.11))
model.add(Dense(50))
model.add(PReLU())

model.add(Dropout(0.09))
model.add(Dense(30))
model.add(PReLU())

model.add(Dropout(0.07))
model.add(Dense(25))
model.add(PReLU())

model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# train model
model.fit(X_train, y_train, batch_size=64, nb_epoch=85,
          validation_data=(X_eval, y_eval), verbose=2, show_accuracy=True)

# generate submission
ids, X = get_test_data()
X = imputer.transform(X)
print('Data shape:', X.shape)
X, scaler = preprocess_data(X, scaler)
preds = model.predict(X, batch_size=256)[:, 1]
with open('output/keras-pyramidal-mlp.csv', 'w') as f:
    f.write('id,Y\n')
    for ID, p in zip(ids, preds):
        f.write('%s,%.8f\n' % (ID, p))
