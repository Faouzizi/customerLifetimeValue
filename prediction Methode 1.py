#########################################################################################################
##########################################     Import some packages    ##################################
#########################################################################################################
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics


#########################################################################################################
##########################################     Import de la data       ##################################
#########################################################################################################
df = pd.read_csv('./data.csv')
colonnes = list(df.columns[:7])
colonnes.append(df.columns[-1])

df = df[colonnes]
#########################################################################################################
##############     On vérifie que toutes les données sont au bon format       ###########################
#########################################################################################################
df.dtypes
df.head()


#########################################################################################################
##############           On effectue une petite analyse de correlation        ###########################
#########################################################################################################



df.drop('customer_id', axis=1).corr()


#########################################################################################################
##############                                    Prédiction                  ###########################
#########################################################################################################
variables_explicatives = df.drop(['customer_id','clv'], axis=1)
variable_cible = df.clv



x_train, x_test, y_train,y_test = train_test_split(variables_explicatives, variable_cible, test_size=0.2)

# build model on training data
model = LinearRegression()
model.fit(x_train, y_train)
model.predict(x_train)

print("Coefficients : \n", model.coef_)
print("Intercept : \n", model.intercept_)



# Test result on testing Data
predictions = model.predict(x_test)
predictions

sklearn.metrics.r2_score(y_test, predictions)

abs(y_test-predictions).mean()
abs(y_test-predictions).median()
sklearn.metrics.mean_squared_error(y_test,predictions)
sklearn.metrics.mean_squared_error(y_train,model.predict(x_train))


results = pd.DataFrame({'y_True':y_test.values, 'y_predicted':predictions})

model.predict(np.array([0,47.99, 6.99,0,47.99, 47.99]).reshape(1, -1) )#822.82


x_test.iloc[0])
