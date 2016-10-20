"""
Helper module for various machine learning utilites and scoring functions.s
"""
from sklearn.externals import joblib


def save_xgb_and_desc(model, model_name, params, score, std,
                      seed, desc_file='models/model_desc.txt'):
    """
    Save a trained XGBoost model to a file
    and log its performance.

    Args:
        model - a trained XGBoost model.
        model_name - file name for the model.
        params - parameters used to train the model.
        score - cross-validation score for the model.
        std - std deviation of the cross-validation score.
        seed - seed used to train the model.
        desc_file - file to log model information.
    """
    model_name = model_name
    model_path = '{}/{}'.format('models', model_name)
    model.save_model(model_path)

    with open('models/model_desc.txt', 'a') as f:
        f.write('Model {0} was trained with the following params:\n{1}\n'
                .format(model_name, params))
        f.write('seed: {}\n'.format(seed))
        f.write('Training cross-validation (mean_score, std_dev):{},{}\n'
                .format(score, std))


def save_model_and_desc(model, model_name, params, score, std,
                      seed, desc_file='models/model_desc.txt'):
    """
    Save a trained scikit-learn model to a file
    and log its performance.

    Args:
        model - a trained sklearn model.
        model_name - file name for the model.
        params - parameters used to train the model.
        score - cross-validation score for the model.
        std - std deviation of the cross-validation score.
        seed - seed used to train the model.
        desc_file - file to log model information.
    """
    model_name = model_name
    model_path = '{}/{}.pkl'.format('models', model_name)
    joblib.dump(model, model_path)

    with open('models/model_desc.txt', 'a') as f:
        f.write('Model {0} was trained with the following params:\n{1}\n'
                .format(model_name, params))
        f.write('seed: {}\n'.format(seed))
        f.write('Training cross-validation (mean_score, std_dev):{},{}\n'
                .format(score, std))


def write_results(outfile, predictions):
    """
    Write a model's predictions to a Kaggle-ready submission file.
    """
    ID = range(49999, 99998 + 1)
    with open(outfile, 'w') as f:
        f.write('id,Y\n')
        for instance, prediction in zip(ID, predictions):
            f.write('{},{}\n'.format(instance, prediction))
