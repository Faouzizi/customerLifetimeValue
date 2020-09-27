##### import pacjages
#from sklearn import model_selection
import pandas as pd
from sklearn.metrics import r2_score
#from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import train_test_split
#import sklearn.metrics
import numpy as np
from sklearn.impute import KNNImputer
import reverse_geocode
#import scipy.stats as stat
#from scipy.stats import boxcox
#import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
#from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
#from sklearn.metrics import mean_squared_error
import joblib
path_model = '/Volumes/Carte_mem/medium/Customer LifeTime Value/customerLifetimeValue/'

def imputation_statique(df, statique):
    ###############################################################
    # Cette fonction vous permettra d'imputer les données manquantes
    # Si statique=True alors l'imputation se fera par la median ou le mode
    # selon le type des données en entrée
    ###############################################################
    missing_data = df.apply(lambda x: np.round(x.isnull().value_counts()*100.0/len(x),2)).iloc[0]
    columns_MissingData = missing_data[missing_data<100].index
    if imputation_statique:
        for col in columns_MissingData:
            if df[col].dtype=='O':
                df[col] = df[col].fillna(df[col].mode().iloc[0])
            else:
                df[col] = df[col].fillna(df[col].median())
    else:
        imputer = KNNImputer(n_neighbors=3)
        ids = df.CustomerID
        X = pd.concat([pd.get_dummies(df.drop('CustomerID', axis=1).select_dtypes('O')), df.drop('CustomerID', axis=1).select_dtypes(exclude='O')], axis=1)
        X_filled_knn = pd.DataFrame(imputer.fit_transform(X))
        X_filled_knn.columns = X.columns
        for col in columns_MissingData:
            print(col)
            if df[col].dtypes=='O':
                df_temp =X_filled_knn.filter(regex='^'+col+'*')
                df_temp.columns = [x.replace(col+'_', '') for x in df_temp.columns]
                df[col] = df_temp.idxmax(1)
            else:
                df[col] = np.round(X_filled_knn[col],2)
    return(df)
##################################################################################################
##############           traitement des valeurs extrèmes #########################################
##################################################################################################
def traiter_valeurs_extremes_continues(df, variable_cible):
    ###############################################################
    # Cette fonction vous permettra de traiter les valeurs extrèmes
    # les valeurs extrèmes seront remplacé par moyenne dans ce cas
    ###############################################################
    for col in df.select_dtypes(exclude='O'):
        if col != variable_cible:
            q1 = df[col].quantile([0.25]).values[0]
            q3 = df[col].quantile([0.75]).values[0]
            IC_valeur_non_aberantes = [q1 - 2*(q3-q1), q3 + 2*(q3-q1)]
            df.loc[df[col]<IC_valeur_non_aberantes[0], col] = df[col].mean()
            df.loc[df[col]>IC_valeur_non_aberantes[1], col] = df[col].mean()
    return(df)


#########################################################################################################
##########################################     Import de la data       ##################################
#########################################################################################################
df = pd.read_csv('/Volumes/Carte_mem/medium/LifeTimeValue/data/CLV_Train.csv')
##################
# Le rôle de l'échantillon ci-dessous est de valider notre modèle sur des observations non vu par le modèle.
df_val = df[-100:]


#########################################################################################################
##########################################     Traitement des données       #############################
#########################################################################################################
def get_data_processed(df, val=False):
    #########################################################################################################
    ##############     On vérifie que toutes les données sont au bon format       ###########################
    #########################################################################################################
    df['Income'] = pd.to_numeric(df['Income'],errors='coerce')
    # on remplace la variable de geolocalisation par la variable country
    coordonnees = tuple(df['Location.Geo'].map(lambda x: tuple(x.split(',')) if x.split(',')[0]!='NA' else (17,77)))
    df['country'] = [x['country'] for x in reverse_geocode.search(coordonnees)]
    # on supprime la varibale de geolocalisation
    df.drop('Location.Geo', axis=1, inplace=True)
    #########################################################################################################
    ##############               On imputeles valeurs manquantes                  ###########################
    #########################################################################################################
    imputation_statique(df, statique='')
    ########################################################################################################
    ##############           TRAIOTEMENT VALEURS EXTREMES        ###########################
    #########################################################################################################
    if val:
        df = traiter_valeurs_extremes_continues(df, variable_cible='Customer.Lifetime.Value')
    else:
        df = traiter_valeurs_extremes_continues(df, variable_cible='sCustomer.Lifetime.Value')
    #########################################################################################################
    ##############                                    Data processing                  ###########################
    #########################################################################################################
    variables_explicatives = df.drop(['CustomerID','Customer.Lifetime.Value'], axis=1)
    variable_cible = df['Customer.Lifetime.Value']
    variables_explicatives['Monthly.Premium.AutoSquare'] = variables_explicatives['Monthly.Premium.Auto']**2
    variables_explicatives['Total.Claim.AmountSquare'] = variables_explicatives['Total.Claim.Amount']**2
    variables_explicatives['Number.of.Open.ComplaintsSquare'] = variables_explicatives['Number.of.Open.Complaints']**2
    variables_explicatives['IncomeSquare'] = variables_explicatives['Income']**2
    variables_explicatives['Months.Since.Last.ClaimSquare'] = variables_explicatives['Months.Since.Last.Claim']**2
    variables_explicatives['Months.Since.Policy.InceptionSquare'] = variables_explicatives['Months.Since.Policy.Inception']**2
    variables_explicatives['Months.Since.Policy.InceptionSquare'] = variables_explicatives['Months.Since.Policy.Inception']**2
    variables_explicatives['Number.of.PoliciesSquare'] = variables_explicatives['Number.of.Policies']**2
    variables_explicatives_continues = variables_explicatives.select_dtypes('float')
    ### One hot encoding
    variables_explicatives_qualitatives = variables_explicatives.select_dtypes('O')
    variables_explicatives_qualitatives = pd.get_dummies(variables_explicatives_qualitatives)
    variables_explicatives = pd.concat([variables_explicatives_continues, variables_explicatives_qualitatives], axis=1)
    ### redimensionnement
    scaler = StandardScaler()
    scaler.fit(variables_explicatives)
    colonnes = variables_explicatives.columns
    variables_explicatives = pd.DataFrame(scaler.transform(variables_explicatives))
    variables_explicatives.columns = colonnes
    return(variables_explicatives, variable_cible)

variables_explicatives, variable_cible = get_data_processed(df)
X, y = variables_explicatives[:-100],variable_cible[:-100]
X_val, y_val = variables_explicatives[-100:], df_val['Customer.Lifetime.Value']

#########################################################################################################
##########################################     Création du modèle       #################################
#########################################################################################################
def build_model(X,y, path):
    xg_reg = xgb.XGBRegressor(objective ='reg:squarederror',
                              colsample_bytree = 0.5,
                              learning_rate = 0.1,
                              max_depth = 4,
                              alpha = 1,
                              n_estimators = 400)
    xg_reg.fit(X,y)
    #save the model
    filename = path+'finalized_model_CLV.sav'
    joblib.dump(xg_reg, filename)
    print(r2_score(y, xg_reg.predict(X)))

#########################################################################################################
##################     Utiliser le modèle pour prédir les valeurs de la clv       #######################
#########################################################################################################
def predict_using_model(X_test, path):
    # load the model from disk
    loaded_model = joblib.load(path)
    result = pd.DataFrame({'pred':loaded_model.predict(X_test)})
    return(result)




#########################################################################################################
##########################################     Création du modèle       #################################
#########################################################################################################
build_model(X,y, path_model)
res = predict_using_model(X_val,path_model+'finalized_model_CLVbis.sav')
res['reel'] = y_val.values
print(np.mean(abs(res.pred-res.reel)))
