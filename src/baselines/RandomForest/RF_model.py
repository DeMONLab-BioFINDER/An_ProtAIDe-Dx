#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Written by Lijun An and DeMON Lab under MIT license:
https://github.com/DeMONLab-BioFINDER/DeMONLabLicenses/blob/main/LICENSE
'''
import copy
from sklearn.ensemble import RandomForestClassifier
from utils.metrics import clf_metrics


class BaseRF:
    def __init__(self) -> None:
        pass


def construct_model_params(hyper_params, nb_est=100):
    """
    Constructing model parameters for random forest.

    Args:
        hyper_params (dict): 
            Hyperparameters for the random forest model.
        nb_est (int, optional): 
            Number of estimators for the random forest. Defaults to 100.

    Returns:
        dict: Model parameters for the random forest.
    """
    if hyper_params is None:
        model_params = {
            'n_estimators':100,
            'max_depth': None,
            'max_features': 'sqrt'}
    else:
        model_params = {
            'n_estimators':nb_est,
            'max_depth': hyper_params['max_depth'],
            'max_features': hyper_params['max_features']}
    return model_params


class RFClfModule(BaseRF):
    """
    Random Forest classifier module.

    Args:
        BaseRF (class): Base class for Random Forest models.
    """
    def __init__(self, 
                 features, 
                 target,
                 nb_est=100,
                 search_range=None, 
                 hyper_params=None):
        """
        Init Random Forest Classifier module.

        Args:
            features (list): List of feature column names.
            target (str): Target column name.
            nb_est (int, optional): 
                Number of estimators for the Random Forest. Defaults to 100.
            search_range (dict, optional): 
                Hyperparameter search range. Defaults to None.
            hyper_params (dict, optional): 
                Hyperparameters for the Random Forest model. Defaults to None.
        """
        super(RFClfModule).__init__()
        self.features = features
        self.target = target
        self.nb_est = nb_est
        if search_range is None:
            # use default
            search_range = {
                'max_depth': [None, 10, 100],
                'max_features': ['sqrt', 'log2']
            }
        self.search_range = search_range
        self.model_params = construct_model_params(
            hyper_params, nb_est)
   
    def train(self, train_df):
        """
        Train the Random Forest classifier.

        Args:
            train_df (pd.DataFrame): Training data.

        Returns:
            RandomForestClassifier: Trained Random Forest classifier.
        """
        # train a Random Forest classifier
        clf = RandomForestClassifier(
            n_estimators=self.model_params['n_estimators'],
            max_depth=self.model_params['max_depth'],
            max_features=self.model_params['max_features'],
            random_state=0
        )
        clf.fit(train_df[self.features],
                train_df[self.target])
        return clf

    def tune(self, train_df, val_df):
        """
        Tune the Random Forest classifier using validation data.

        Args:
            train_df (pd.DataFrame): Training data.
            val_df (pd.DataFrame): Validation data.

        Returns:
            tuple: Best model and best hyperparameters.
        """
        # run hyperparameter tuning
        best_max_depth = None
        best_max_features = 'auto'
        best_acc = 0
        best_model = None
        for max_dep in self.search_range['max_depth']:
            for max_fea in self.search_range['max_features']:
                clf = RandomForestClassifier(
                    n_estimators=self.model_params['n_estimators'],
                    max_depth=max_dep,
                    max_features=max_fea,
                    random_state=0
                )
                clf.fit(train_df[self.features],
                        train_df[self.target])
                # make prediction on validation set
                val_pred = clf.predict(val_df[self.features])
                # evaluation
                acc = clf_metrics(
                    val_df[self.target].values.reshape((-1, )),
                    val_pred.reshape((-1, )),
                    nb_class=2,
                    threshold=0.5
                )['bas']
                if acc > best_acc:
                    best_acc = acc
                    best_max_depth = max_dep
                    best_max_features = max_fea
                    best_model = copy.deepcopy(clf)
                

        best_hyperparam = {'max_depth': best_max_depth, 
                           'max_features': best_max_features}
        return best_model, best_hyperparam

    def predict(self,
                model,
                df):
        """
        Make predictions using the trained Random Forest classifier.

        Args:
            model (RandomForestClassifier): Trained Random Forest classifier.
            df (pd.DataFrame): Data for making predictions.
        Returns:
            np.ndarray: Predicted labels.
        """
        pred = model.predict(df[self.features])
        return pred