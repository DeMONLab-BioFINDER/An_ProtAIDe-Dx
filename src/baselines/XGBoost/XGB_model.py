#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Written by Lijun An and DeMON Lab under MIT license:
https://github.com/DeMONLab-BioFINDER/DeMONLabLicenses/blob/main/LICENSE
'''
from src.baselines.XGBoost.XGB_utils import\
    model_fit, grid_search, gen_xgb_input, model_inference,\
    construct_model_params


class BaseXGB:
    """
    Base class for XGBoost models.
    """
    def __init__(self) -> None:
        pass


class XGBModule(BaseXGB):
    """
    XGBoost model module.

    Args:
        BaseXGB (_type_): _description_
    """
    def __init__(self, 
                 features, 
                 target, 
                 task, 
                 nb_class=1,
                 nb_boost=100,
                 search_range=None, 
                 hyper_params=None):
        """
        Initialize the XGBoost model module.

        Args:
            features (list): List of feature names.
            target (str): Target variable name.
            task (str): Task type ('binary', 'multi', etc.).
            nb_class (int, optional): 
                Number of classes for multi-class classification. 
                Defaults to 1.
            nb_boost (int, optional): 
                Number of boosting rounds. Defaults to 100.
            search_range (dict, optional): 
                Hyperparameter search range. Defaults to None.
            hyper_params (dict, optional): 
                Hyperparameters for the model. Defaults to None.

        Raises:
            ValueError: 
                If multi-class classification is selected 
                with fewer than 3 classes.
        """
        super(XGBModule).__init__()
        self.features = features
        self.target = target
        self.task = task
        self.nb_boost = nb_boost
        if search_range is None:
            # use default
            search_range = {
                'max_depth': [3, 5, 7],
                'subsample': [0.6, 0.8, 1]
            }
            if task == 'binary':
                search_range['threshold'] = [0.1, 0.2, 0.3, 0.4, 0.5]
            if task == 'multi' and nb_class < 3:
                raise ValueError(
                    "Multi-classification needs at least 3 classes!")
        self.search_range = search_range
        self.model_params = construct_model_params(hyper_params, 
                                                   task,
                                                   nb_class)
   
    def train(self, train_df):
        """
        Train the XGBoost model.

        Args:
            train_df (pd.DataFrame): Training data.

        Returns:
            model: Trained XGBoost model.
        """
        xg_train = gen_xgb_input(train_df, self.features, self.target)
        return model_fit(xg_train, self.model_params, self.nb_boost)

    def tune(self, 
             train_df, 
             val_df):
        """
        Tune the XGBoost model.

        Args:
            train_df (pd.DataFrame): Training data.
            val_df (pd.DataFrame): Validation data.
        Returns:
            tuple: Best model and best hyperparameters.
        """
        best_model, best_hyperparam = grid_search(
            train_df, val_df, self.features,
            self.target, self.model_params,
            self.nb_boost, self.task, self.search_range)
        return best_model, best_hyperparam


    def predict(self, 
                model, 
                df):
        """
        Predict using the XGBoost model.

        Args:
            model: Trained XGBoost model.
            df (pd.DataFrame): Data for prediction.

        Returns:
            pd.Series: Predicted values.
        """
        pred = model_inference(model, df, self.features)
        return pred