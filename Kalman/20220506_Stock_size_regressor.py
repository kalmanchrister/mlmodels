"""Optimal regression found to estimate catch sizes."""
import pandas as pd
import math
# import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
# from sklearn.pipeline
# from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xgboost as xgb
# from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import shap
import time

path_str = 'R:\\Ráðgjöf\\Maris Optimum/gr_data\\'
# path_str = ''


def get_new_data(fractile):
    """
    Fetch all data for the regression.

    Parameters
    ----------
    fractile : integer.

    Returns
    -------
    XX_df : Dataframe with data for all the dependent variables.
    YY : Dataframe with data for the the independent variable.

    """
    X100_df = pd.read_csv(path_str + 'distribution100.csv',
                          sep=",")
    ysq_df = X100_df[['ar', 'max(cum)']].copy()
    ysq_df.set_index(['ar'], inplace=True)
    ysq_df = ysq_df[~ysq_df.index.duplicated(keep='first')]

    X_df = pd.read_csv(path_str + 'distribution' + fractile + '.csv',
                       sep=",")

    catch_df = pd.read_csv(path_str + 'golden_redfish_catch.csv',
                           sep=";")

    catch_df.at[37, 'year'] = 2022.0
    catch_df.at[37, 'catch'] = 26
    catch_df.at[37, 'number'] = 29

    X_cal_df = pd.read_csv(path_str + 'distribution_commercial.csv',
                           sep=",")
    X_cal_df.drop(1605, axis=0, inplace=True)

    X_cal_df = X_cal_df.pivot(index='ar',
                              columns='lengd',
                              values='per_length')
    X_cal_df = X_cal_df.fillna(0)
    X_cal_df.columns = 1000 + X_cal_df.columns
    X_cal_df.columns = X_cal_df.columns.astype(int).astype(str)

    catch_df = catch_converter(X_cal_df, catch_df)
    catch_df.year = catch_df.year.astype(int)
    catch_df.set_index(catch_df.year, inplace=True)

    X_cal_df = X_cal_df.mul(catch_df.number*-1e6, axis=0)

    XX_df = X_df.pivot(index='ar',
                       columns='lengd',
                       values='per_length')

    XX_df = pd.merge(XX_df, ysq_df, right_index=True, left_index=True)

    XX_df.drop(11.9, axis=1, inplace=True)
    XX_df.drop(12.5, axis=1, inplace=True)
    XX_df.drop(12.6, axis=1, inplace=True)
    XX_df.drop(13.1, axis=1, inplace=True)
    XX_df.drop(13.4, axis=1, inplace=True)
    XX_df.drop(13.6, axis=1, inplace=True)
    XX_df.drop(13.7, axis=1, inplace=True)
    XX_df.drop(13.9, axis=1, inplace=True)
    XX_df.drop(14.4, axis=1, inplace=True)
    XX_df.drop(14.5, axis=1, inplace=True)
    XX_df.drop(14.7, axis=1, inplace=True)
    XX_df.drop(14.8, axis=1, inplace=True)
    XX_df.drop(14.9, axis=1, inplace=True)

    XX_df.columns = XX_df.columns.astype(str)

    YX = pd.read_csv(path_str+"RED_numbers_at_age.csv", sep=";")
    YY = YX.iloc[15:53, 28]
    
    YZ = pd.read_csv("R:\\Ráðgjöf\\Maris Optimum\\stock_measurement\\stock_measurement.csv", sep=";")
    YY = YZ.iloc[:,1]


    # XX_df = XX_df.join(X_cal_df.iloc[:, :])

    XX_df = XX_df.join(catch_df.loc[:, 'number'])
    XX_df.index = XX_df.index.astype(str)

    '''s = XX_df.index[29:35]
    s.index = XX_df[29:35]
    s = pd.get_dummies(s)
    s.index = XX_df.index[29:35]
    XX_df = XX_df.join(s)'''

    XX_df = XX_df.fillna(0)
    heat_maps(XX_df, YY)
    return (XX_df, YY)

def stock_weight_converter(X_catch_per_df, catch_df): 
    """
    Stock_weight information used to formulate and calculate units from kg.

    Parameters
    ----------
    X_catch_per_df : Dataframe containing percentages of catch.
    stoxk_df : Dataframe containing total stock in kg.

    Returns
    -------
    catch_df : Dataframe containing total catch in lengths.
    """
    path_grs_str = 'R:/Ráðgjöf/Maris Optimum/Golden_redfish_model/'
    path_grs_weight =\
        'R:/Ráðgjöf/Maris Optimum/Data_from_Hafro_20220906/ssb_weights/'

    red_length_weights = pd.read_csv(
        path_grs_weight + "red_length_weights.csv")
    red_length_weights.set_index('ar', inplace=True)

    wl_df = pd.read_csv(path_grs_str+'RED_gadget_n_at_age.csv', sep=',')

    Xl = wl_df[['year', 'mean_length']].copy()
    yl = wl_df[['year', 'mean_weight']].copy()

    for index, row in Xl.iterrows():
        Xl.at[index, 'squared'] = row[1]**2

    for year in range(1985, 2022):
        average_weight = 0
        reg_X = Xl[Xl['year'] == year]
        reg_y = yl[yl['year'] == year]
        reg = LinearRegression().fit(
            reg_X[['mean_length', 'squared']], reg_y[['mean_weight']])
        b = reg.coef_[0][0]
        a = reg.coef_[0][1]
        c = reg.intercept_
        for col in range(1020, 1060):
            if year > 1995 and \
               pd.isna(red_length_weights.loc[year, str(col-1000)]) is False:
                average_weight += 1/1000*(X_catch_per_df.loc[year, str(col)]) * red_length_weights.loc[year, str(col-1000)]
            else:
                average_weight += (X_catch_per_df.loc[year, str(col)]) * (
                    a*(col - 1000)**2 + b*(col - 1000) + c)
        catch_df.at[
            year - 1985, 'number'] = catch_df.loc[
                year - 1985, 'catch']/average_weight

    df = red_length_weights
    scaled_df = (df - df.min(axis=0))/(df.max(axis=0) - df.min(axis=0))
    ax = sns.heatmap(scaled_df, linewidths=.5, cmap='RdYlGn')
    ax.set(ylabel='year')
    plt.show()
    ax=sns
    return catch_df

def catch_converter(X_catch_per_df, catch_df): 
    """
    Catch information used to formulate and calculate units from kg.

    Parameters
    ----------
    X_catch_per_df : Dataframe containing percentages of catch.
    catch_df : Dataframe containing total cath in kg.

    Returns
    -------
    catch_df : Dataframe containing total catch in lengths.
    """
    path_grs_str = 'R:/Ráðgjöf/Maris Optimum/Golden_redfish_model/'
    path_grs_weight =\
        'R:/Ráðgjöf/Maris Optimum/Data_from_Hafro_20220906/ssb_weights/'

    red_length_weights = pd.read_csv(
        path_grs_weight + "red_length_weights.csv")
    red_length_weights.set_index('ar', inplace=True)

    wl_df = pd.read_csv(path_grs_str+'RED_gadget_n_at_age.csv', sep=',')

    Xl = wl_df[['year', 'mean_length']].copy()
    yl = wl_df[['year', 'mean_weight']].copy()

    for index, row in Xl.iterrows():
        Xl.at[index, 'squared'] = row[1]**2

    for year in range(1985, 2022):
        average_weight = 0
        reg_X = Xl[Xl['year'] == year]
        reg_y = yl[yl['year'] == year]
        reg = LinearRegression().fit(
            reg_X[['mean_length', 'squared']], reg_y[['mean_weight']])
        b = reg.coef_[0][0]
        a = reg.coef_[0][1]
        c = reg.intercept_
        for col in range(1020, 1060):
            if year > 1995 and \
               pd.isna(red_length_weights.loc[year, str(col-1000)]) is False:
                average_weight += 1/1000*(X_catch_per_df.loc[year, str(col)]) * red_length_weights.loc[year, str(col-1000)]
            else:
                average_weight += (X_catch_per_df.loc[year, str(col)]) * (
                    a*(col - 1000)**2 + b*(col - 1000) + c)
        catch_df.at[
            year - 1985, 'number'] = catch_df.loc[
                year - 1985, 'catch']/average_weight

    df = red_length_weights
    scaled_df = (df - df.min(axis=0))/(df.max(axis=0) - df.min(axis=0))
    ax = sns.heatmap(scaled_df, linewidths=.5, cmap='RdYlGn')
    ax.set(ylabel='year')
    plt.show()
    return catch_df

def heat_maps(X, y):
    """
    Plot heatmaps.

    Parameters
    ----------
    X : Independent variables.
    y : Dependent variables.

    Returns
    -------
    Pictues of heat maps.

    """
    sns.color_palette("vlag", as_cmap=True)
    ax = sns.heatmap(X.iloc[:, :50], cmap='RdYlGn')
    ax.set(ylabel='year')
    plt.figure(figsize=(6, 4))
    plt.show()
    

    ax = sns.heatmap((
        y.to_numpy() * X.iloc[:, 0:50].T).T, cmap='RdYlGn')
    ax.set(ylabel='year')
    plt.figure(figsize=(6, 4))
    plt.show()

def fitting_plot(y_test, y_pred_test, X_test, xgb_regressor):
    """
    Plot how well predictions fit the independent test variables.

    Parameters
    ----------
    y_test : independent y variables.
    y_pred_test : independent predictions.
    X_test : datframe containing the dependent test variables.
    xgb_regressor : the optimal regressor.

    Returns
    -------
    None.

    """
    x_ax = range(len(y_test))
    plt.scatter(x_ax,
                y_test,
                s=5,
                color="blue",
                label="original")
    plt.plot(x_ax,
             y_pred_test,
             lw=0.8,
             color="red",
             label="predicted")
    plt.xticks(x_ax,
               X_test.index,
               rotation=45)
    plt.grid(True)
    plt.legend()
    plt.show()
    timestr = time.strftime("%Y%m%d-%H%M%S")

def error_plot(regressor):
    """
    Plot the accuracy of the model.

    Parameters
    ----------
    regressor : Xgboost regressor.

    Returns
    -------
    None.

    """
    results = regressor.evals_result()
    epochs = len(results['validation_0']['mae'])
    x_axis = range(0, epochs)
    fig, ax = plt.subplots()
    plt.grid(True, which='major')
    ax.plot(x_axis, results['validation_0']['mae'],
            label='Train')
    ax.plot(x_axis, results['validation_1']['mae'],
            label='Test')
    ax.legend()
    plt.ylabel('Error')
    plt.title('Error')
    plt.show()
    timestr = time.strftime("%Y%m%d-%H%M%S")


def shap_calculations_xgb(regressor, XX_df):
    """
    Calculate shap vales for the xgb_regressor and the dependent variables.

    Parameters
    ----------
    regressor : Xgboost regressor.
    XX_df : Dependent variables.

    Returns
    -------
    None.

    """
    explainer = shap.TreeExplainer(regressor)

    shap_values = explainer.shap_values(XX_df)
    plt.set_cmap("plasma")
    shap.summary_plot(shap_values,
                      XX_df,
                      plot_type="violin",
                    
                      max_display=10, 
                      show=False)
    
    #plt.xlabel("Mikilvægi óháðra breyta til hækkunar (vinstra megin) og lækkunar (hægra megin)")
    plt.ylabel("Nöfn óháðu breytanna, (lengdir í cm, fjöldi fiska í rallinu og magn af veiði)")
    plt.xlabel("Tonn")
    #plt.yticks(fontsize=2)
    plt.gcf().axes[-1].set_ylabel('Rauður litur þýðir lækkun, blár litur þýðir hækkun')
    plt.gcf().axes[-1].set_aspect(100)
    plt.gcf().axes[-1].set_box_aspect(100)
    plt.gcf().axes[-1 ].set_yticklabels(['Lágt', 'Hátt'])
    plt.show()
    
    shap_values = explainer(XX_df)
    shap.waterfall_plot(shap_values[34], max_display=15)
    f = shap.plots.force(explainer.expected_value, shap_values.values, XX_df)
    shap.save_html("index.htm", f)
    shap.plots.scatter(shap_values[:, 51])
    plt.show
    shap_interaction_values = shap.TreeExplainer(
        regressor).shap_interaction_values(XX_df.iloc[:2000, :])
    shap.dependence_plot(
        ("max(cum)", "number"),
        shap_interaction_values,
        XX_df)
    plt.show


def svr_regression(X_train, y_train, X_test, y_test):
    """
    Regress using support vector machine.

    Parameters
    ----------
    X_train : Dependent training variables.
    y_train : Independent training variables.
    X_test : Dependent testing variables.
    y_test : Independent testing variables.

    Returns
    -------
    Prediction for test variables.

    """
    pipe_svr = make_pipeline(
        StandardScaler(),
        SVR())

    param_range = [.01, .1, 10, 100]
    param_grid = [{
        'svr__C': param_range,
        'svr__kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
        'svr__epsilon': param_range,
        'svr__coef0': param_range,
        'svr__gamma': param_range}]

    svr_regressor = GridSearchCV(estimator=pipe_svr,
                                 param_grid=param_grid,
                                 scoring='neg_mean_absolute_error',
                                 n_jobs=-1,
                                 verbose=1)

    svr_regressor.fit(X_train, y_train)
    print(svr_regressor.best_estimator_)
    return svr_regressor.predict(X_test)


def regression_over_possible_values_XGB(X, y, interval_int, parameters):
    """
    Loop over possible stock sizes, regressing in every step.

    Parameters
    ----------
    X : Dataframe with dependent variables.
    y : Dataframe with independent variables.
    interval_int : step interval.


    Returns
    -------
    result_dict : json string with solutions in each interval.

    """
    test_size = .20
    seed = 5
    result_dict = {'fjoldi2022': [], 'fjoldi2021': [],
                   'fjoldi2020': [], 'mae': [], 'rmse': [], 'r2': [],
                   'evs': [], 'regressor':[]}

    xgb1 = xgb.XGBRegressor(objective='reg:squarederror', seed=seed)

    for add_int in range(0, 110000, interval_int):
        print(add_int)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=seed)

        n_iter = 200
        n_iter = n_iter

        xgb_regressor = GridSearchCV(xgb1,
                                     parameters,
                                     n_jobs=-1,
                                     cv=3,
                                     verbose=1)

        eval_set = [(X_train, y_train), (X_test, y_test)]

        xgb_regressor.fit(X_train,
                          y_train,
                          eval_set=eval_set,
                          verbose=False)

        y_pred_test = xgb_regressor.predict(X_test)

        # y_pred_test = svr_regression(X_train, y_train, X_test, y_test)

        result_dict['fjoldi2022'].append(y.iloc[37])
        result_dict['fjoldi2021'].append(y.iloc[36])
        result_dict['fjoldi2020'].append(y.iloc[35])
        result_dict['regressor'].append(xgb_regressor)

        result_dict['mae'].append(mean_absolute_error(y_test,
                                                      y_pred_test))
        result_dict['rmse'].append(math.sqrt(mean_squared_error(y_test,
                                                                y_pred_test)))
        result_dict['r2'].append(r2_score(y_test,
                                          y_pred_test))
        result_dict['evs'].append(explained_variance_score(y_test,
                                                           y_pred_test))
        y.iat[35] += interval_int * (y.iloc[35]/y.iloc[36])
        y.iat[36] += interval_int * (y.iloc[36]/y.iloc[37])
        y.iat[37] += interval_int

    min_value = min(result_dict['mae'])
    min_index = result_dict['mae'].index(min_value)

    y.iat[35] = result_dict['fjoldi2020'][min_index]
    y.iat[36] = result_dict['fjoldi2021'][min_index]
    y.iat[37] = result_dict['fjoldi2022'][min_index]

    regressor = GridSearchCV(xgb1,
                             parameters,
                             n_jobs=-1,
                             cv=3,
                             verbose=0)

    regressor.fit(X, y)
    params = regressor.best_params_
    regressor = xgb.XGBRegressor(**params)
    regressor.fit(X, y)

    shap_calculations_xgb(regressor, X)
    
    print(regressor.predict(X))

    return result_dict

# %%
def free_regression_XGB(X, y, parameters, start_year, end_year):
    """
    Regression where part of y values are free.

    Parameters
    ----------
    X : Dataframe with dependent variables.
    y : Dataframe with independent variables.

    Returns
    -------
    returns array with predection for y based intiial values of X and
    the regression.

    """
    seed = 3

    xgb1 = xgb.XGBRegressor(objective='reg:squarederror',
                            seed=seed)



    X_train = pd.concat([X.iloc[:start_year, :], X.iloc[end_year:, :]])
    y_train = pd.concat([y.iloc[:start_year], y.iloc[end_year:]])
    X_test = X.iloc[start_year:end_year, :]
    y_test = y.iloc[start_year:end_year]

    eval_set = [(X_train, y_train), (X_test, y_test)]

    regressor = GridSearchCV(xgb1,
                             parameters,
                             n_jobs=-1,
                             cv=3,
                             verbose=False)

    regressor.fit(X_train,
                  y_train,
                  eval_set=eval_set,
                  verbose=1)

    # params = regressor.best_params_
    # regressor = xgb.XGBRegressor(**params)
    # regressor.fit(X, y)
    fitting_plot(y_test,
                 regressor.predict(X_test),
                 X_test,
                 regressor)

    return regressor.predict(X)
# %%
def slack_XGB(X, y, parameters):
    """
    Regression where part of y values are free.

    Parameters
    ----------
    X : Dataframe with dependent variables.
    y : Dataframe with independent variables.

    Returns
    -------
    returns array with predection for y based intiial values of X and
    the regression.

    """
    seed = 5

    xgb1 = xgb.XGBRegressor(objective='reg:squarederror',
                            seed=seed)


    regressor = GridSearchCV(xgb1,
                             parameters,
                             n_jobs=-1,
                             cv=3,
                             verbose=False)

    regressor.fit(X,
                  y,
                  verbose=False)

    error_plot(regressor)
    # params = regressor.best_params_
    # regressor = xgb.XGBRegressor(**params)
    # regressor.fit(X, y)

    return regressor.predict(X)
# %%
def plot_result_range(result_dict, interval_int, fractile, y_offset):
    """
    Plot mae and r^2 for the range of the looped regression.

    Parameters
    ----------
    result_dict : a dictianary containing the results from different regression
    over the interval.
    interval_int : integer with the increments used over the interval.
    fractile : fractile used for Winsorization.


    Returns
    -------
    None.

    """
    result_dict['x'] = range(379+y_offset,
                             489+y_offset,
                             int(interval_int/1e3))
    fig, ax = plt.subplots()
    sns.set(style='whitegrid',
            palette='pastel', )
    sns.lineplot(x='x',
                 y='r2',
                 data=result_dict,
                 color="red",
                 ax=ax)
    ax.set(xlabel='Þúsundir tonna',
           ylabel='Fervik sem skýrast af óháðum breytum, r2,  eru sýnd með rauðum ferli',
           # title='Leiðrétting Winsor notar '+fractile+' hlutfallsbrot.',
           ylim=(0, 1))

    ax2 = ax.twinx()
    sns.lineplot(x='x',
                 y='mae',
                 data=result_dict,
                 color='blue',
                 markers=True, ax=ax2)
    ax2.set(ylabel='Meðaltölugildi skekkjunnar er sýnt með bláum ferli')
    plt.show()
    timestr = time.strftime("%Y%m%d-%H%M%S")
# %% 
def minimum_error_stock_assessment():
    
    minimum_assessment_difference= 1000000
    minimum_start_year = -1
    minimum_end_year = -1
    fractile = "094"
    parameters_free_regression = {
        'eval_metric': ["mae"],
        'learning_rate': [0.05, .1, .3],
        'max_depth': [2],
        'min_child_weight': [2],
        'subsample': [.5, 1],
        'colsample_bytree': [.5, 1],
        'n_estimators': [50]
    }
    for fractile in ("094","096","098"):
        for start_year in range(20,24):
            for end_year in range (30,34):
                
                (X_observed, y_observed) = get_new_data(fractile)
                y_2022 = y_observed.iloc[37]
                
          
                
                y_calculated = pd.Series(free_regression_XGB(X_observed, 
                                                             y_observed, 
                                                             parameters_free_regression,
                                                             start_year,
                                                             end_year))
                
                
                y_offset = int((y_calculated.iloc[37]-y_2022)/1e6)
                y_observed.index=range(0,38)
                y_df = pd.concat([y_observed, 
                                  y_calculated], 
                                 axis=1, 
                                 join='inner')
                y_df.columns = ['Formal stock assessment', 'Machine learning stock assessment']
                y_df['Difference'] = y_df['Formal stock assessment'] - y_df['Machine learning stock assessment']
                y_df = y_df.rename(index = lambda x: x + 1985)
                assessment_difference = sum(y_df['Difference'].abs())/(end_year-start_year)
                if assessment_difference < minimum_assessment_difference:
                    minimum_assessment_difference = assessment_difference
                    minimum_start_year = start_year
                    minimum_end_year = end_year
    return (minimum_start_year, minimum_end_year, minimum_assessment_difference, fractile )
# %%
# def main():
"""
Run main algorithms.

Returns
-------
None.

"""
# runs regression where data has been adjusted for large catches
fractile = '096'
interval_int = 1000

parameters = {
    'eval_metric': ["mae"],
    'learning_rate': [0.05, .1, .3],
    'max_depth': [2],
    'min_child_weight': [2],
    'subsample': [.5],
    'colsample_bytree': [.5],
    'n_estimators': [50]
}


(X_observed, y_observed) = get_new_data(fractile)
result_dict_gb = regression_over_possible_values_XGB(X_observed,
                                                     y_observed,
                                                     interval_int, 
                                                     parameters)
plot_result_range(result_dict_gb,
                  interval_int,
                  fractile,
                  0)

# %% free regression where data has been adjusted for large catches
fractile = "096"       
interval_int = 1000

(X_observed, y_observed) = get_new_data(fractile)
y_2022 = y_observed.iloc[37]

parameters_free_regression = {
    'eval_metric': ["mae"],
    'learning_rate': [.1, .3],
    'max_depth': [2],
    'min_child_weight': [1],
    'subsample': [.5],
    'colsample_bytree': [.5],
    'n_estimators': [50]
}

start_year = 23
end_year = 33

y_calculated = pd.Series(free_regression_XGB(X_observed, 
                                             y_observed, 
                                             parameters_free_regression,
                                             start_year,
                                             end_year))


y_offset = int((y_calculated.iloc[37]-y_2022)/1e6)
y_observed.index=range(0,38)
y_df = pd.concat([y_observed, 
                  y_calculated], 
                 axis=1, 
                 join='inner')
y_df.columns = ['Formal stock assessment', 'Machine learning stock assessment']
y_df['Difference'] = y_df['Formal stock assessment'] - y_df['Machine learning stock assessment']
y_df = y_df.rename(index = lambda x: x + 1985)
y_df['mae'] = ([31982 for x in range(len(y_df.index))])
y_df['mae-'] = - y_df['mae']
y_df['mae++'] =y_df['Machine learning stock assessment'] + y_df['mae'] 
y_df['mae--'] = y_df['Machine learning stock assessment'] - y_df['mae'] 


pd.options.display.float_format = '{:,.0f}'.format
y_df['Formal stock assessment'] = y_df['Formal stock assessment'].astype(float)
y_df.columns = ['Stofnmæling', 'Vélanámsaðferð', 'Mismunur',
                'Undir skekkja', 'Yfir skekkja', 'Efri skekkjumörk', 'Neðri skekkjumörk']
y_df.iloc[13:38,:].plot(kind='line') 
sns.lineplot( y_df.iloc[13:38,:])
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.ylabel("Tonn")
plt.show()

print(y_df.iloc[start_year:38, :])

result_dict_gb = regression_over_possible_values_XGB(X_observed,
                                                     y_calculated,
                                                     interval_int, 
                                                     parameters_free_regression)
plot_result_range(result_dict_gb,
                  interval_int,
                  fractile,
                  y_offset)



# %% pca analysis



scaler = StandardScaler()
X_sca = scaler.fit_transform(X_observed)
pca = PCA(n_components=18)
pca.fit(X_sca)
print((pca.explained_variance_ratio_))
print(pca.singular_values_)

#if __name__ == '__main__':
#    main()