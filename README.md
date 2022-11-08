## mlmodels using Xgboost.

As any Boosting Machine has a tendency of overfitting, XGBoost has an intense focus on addressing "bias-variance trade-off" and facilitates the users to apply a variety of regularization techniques through hyperparameter tuning.

In developing these models the main focus has been on hyperparameter tuning of the native XGBoost API in order to improve its regression performance while addressing "bias-variance trade-off" - especially to alleviate the risk of overfitting.  In order to conduct hyperparameter tuning, this analysis uses the grid search method. In other words, we select the search grid for hyperparameters and calculate the model performance over all the hyperparameter datapoints on the search-grid. Then, we identify the global local minimum of the performance - or the hyperparameter datapoint which yields the best performance (the minimum value of the Loss Function) - as the best hyperparameter values for the tuned model.

This process has been itearated over a range of possible values.

Grid search is time consuming. The jupyter notebook contained within has been provided with streamlined set of "likely" hyperparameters to speed up its running time. This streamlining took a substantial effort. The current notebook will run rather quickly on x64 based processor. 
