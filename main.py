import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics, linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# nacitajme data
data = pd.read_excel('data/flight1/Data_Train.xlsx')

# CHYBAJUCE HODNOTY
# chybajuce hodnoty
# mame chybajuce hodnoty?
#print(data.isnull().values.any())
# kolko?
#print(data.isnull().sum())
# ktore su to? su rovnake -> ano
#print("do they have the same index? -->", data[data["Route"].isnull()].index[0] == data[data["Total_Stops"].isnull()].index[0])
# zistime indexy chybajucich zaznamv
miss_row = data[data["Total_Stops"].isnull()].index[0]
#print("row number:", miss_row)
#vymazeme tento riadok
data = data.drop(miss_row)

# skuska: chybaju este nejake zaznamy?
#print(data.isnull().sum())

# TRANSFORMACIA DAT
# priprava datumu odletu
data['Day'] = pd.to_datetime(data["Date_of_Journey"], format='%d/%m/%Y').dt.day
data['Month'] = pd.to_datetime(data["Date_of_Journey"], format='%d/%m/%Y').dt.month
data['Year'] = pd.to_datetime(data["Date_of_Journey"], format='%d/%m/%Y').dt.year
data.drop('Date_of_Journey', axis=1, inplace=True)

# priprava casov Depature
data['Dep_hour'] = pd.to_datetime(data['Dep_Time'], format='%H:%M').dt.hour
data['Dep_min'] = pd.to_datetime(data['Dep_Time'], format='%H:%M').dt.minute
data.drop('Dep_Time', axis=1, inplace=True)

# priprava casov Arrival
hours, minutes = [], []
for time in data['Arrival_Time']:
    hour_match = re.search(r'(\d{2}):', time)
    if hour_match:
        hours.append(hour_match.group(1))
    minute_match = re.search(r':(\d{2})', time)
    if minute_match:
        minutes.append(minute_match.group(1))

data['Arrival_hour'] = pd.Series(hours)
data['Arrival_min'] = pd.Series(minutes)
data.drop('Arrival_Time', axis=1, inplace=True)

# predpriprava Duration
hours_list, minutes_list = [], []
for time in data["Duration"]:
    hours = minutes = 0
    if 'h' in time:
        hour_index = time.index('h')
        hours = int(time[:hour_index])
    if 'm' in time:
        minutes_index = time.index('m')
        if 'h' in time:
            minutes = int(time[hour_index + 1:minutes_index])
        else:
            minutes = int(time[:minutes_index])
    hours_list.append(hours)
    minutes_list.append(minutes)

data['Duration_hours'] = pd.Series(hours_list)
data['Duration_mins'] = pd.Series(minutes_list)
data.drop('Duration', axis=1, inplace=True)


# konvertujme počet zastavok zo string na int
conditions = [
    data['Total_Stops'].eq('1 stop'),
    data['Total_Stops'].eq('2 stops'),
    data['Total_Stops'].eq('3 stops'),
    data['Total_Stops'].eq('4 stops'), ]

data['Stops'] = np.select(conditions, [1,2, 3, 4], default=0)
data.drop('Total_Stops', axis=1, inplace=True)

# zahodime stare stlpce
data.drop(['Route','Additional_Info'], inplace=True, axis=1)


### OTAZKY ###

# Líši sa cena letenky na základe leteckej spoločnosti?
# predpriprava dat
df = data.copy(deep=True)
df.drop(df[(df["Airline"] == "Multiple carriers") | (df["Airline"] == "Multiple carriers Premium economy") |
           (df["Airline"] == "Jet Airways Business") | (df["Airline"] == "Vistara Premium economy") ].index, inplace = True)

data["Price"] = data["Price"] / 100
df["Price"] = df["Price"] / 100

df = df.drop(10682)
df = df.drop(6474)

def cena_vs_aerolinka():
    # 1. - pozrime si priemery cien leteniek
    avg_prices = df.groupby('Airline')['Price'].mean().sort_values(ascending=False)
    plt.figure(1, figsize=(12, 6))
    avg_prices.plot(kind='bar', color=sns.color_palette("husl", 9))
    plt.title('Priemerná cena letenky leteckej spoločnosti')
    plt.xlabel('Letecká spoločnosť')
    plt.ylabel('Priemerná cena')
    plt.xticks(rotation=30)

    # 2. - priemerna cena za hodinu letu spolocnosti
    df["Average_price_per_hour"] = df["Price"] / df["Duration_hours"]
    price_per_h = df.groupby('Airline')["Average_price_per_hour"].mean()#.replace(np.inf, np.nan)
    #sum_prices = df.groupby('Airline')['Price'].sum()
    #sum_duration = df.groupby('Airline')['Duration_hours'].sum()
    #price_per_h = sum_prices/sum_duration
    plt.figure(2, figsize=(12, 8))
    price_per_h.sort_values(ascending=False).plot(kind='bar', color=sns.color_palette("husl", 9))
    plt.title("Priemerná cena letenky leteckej spoločnosti za hodinu letu")
    plt.xlabel("Letecká spoločnosť")
    plt.ylabel("Priemerná cena / h")
    plt.xticks(rotation=30)
    plt.show()

#cena_vs_aerolinka()


def cena_vs_cas():
    # Líši sa cena letenky na základe času odletu/priletu?
    plt.figure(figsize=(20,8))
    plt.subplot(1,2,1)
    sns.boxplot(x=data["Dep_hour"], y=data["Price"])
    plt.title("Čas odletu vs cena", fontsize=20)
    plt.xlabel("Čas odletu", fontsize=15)
    plt.ylabel("Cena", fontsize=15)

    plt.subplot(1,2,2)
    sns.boxplot(x= df["Arrival_hour"], y=data["Price"])
    plt.title("Čas príletu vs cena", fontsize=20)
    plt.xlabel('Čas príletu', fontsize=15)
    plt.ylabel('Cena', fontsize=15)
    plt.show()

#cena_vs_cas()


def cena_vs_miesto():
    # Líši sa cena letenky od miesta odetu/príletu?
    plt.figure(figsize=(20,8))
    plt.subplot(1,2,1)
    sns.boxplot(x=data["Source"], y=data["Price"])
    plt.title("Miesto odletu vs cena", fontsize=20)
    plt.xlabel("Miesto odletu", fontsize=15)
    plt.ylabel("Cena", fontsize=15)

    plt.subplot(1,2,2)
    sns.boxplot(x= df["Destination"] , y=data["Price"])
    plt.title("Miesto príletu vs cena", fontsize=20)
    plt.xlabel('Miesto príletu', fontsize=15)
    plt.ylabel('Cena', fontsize=15)
    plt.show()

#cena_vs_miesto()


# pozrime korelcanu maticu numerickych premennych
def korelacna_matica():
    df2 = data[["Price", "Day", "Month", "Dep_hour", "Dep_min", "Arrival_hour", "Arrival_min", "Duration_hours", "Duration_mins", "Stops"]]
    plt.figure(figsize=(10, 12))
    sns.heatmap(df2.corr(), vmax=1, cmap="YlGnBu", annot=True, xticklabels=True, yticklabels=True)
    plt.show()

#korelacna_matica()


# premenime kategoricke na numericke premenne a zahodime stare stlpce
df_airline = pd.get_dummies(data['Airline'])
df_source = pd.get_dummies(data['Source'])
df_destination = pd.get_dummies(data['Destination'])

data = pd.concat([data,df_airline, df_source, df_destination],axis=1)
data.drop(['Airline','Source', 'Destination'], inplace=True, axis=1)


### PREDIKCIE ###

data = data.drop(10682)
X_train, X_test, y_train, y_test = train_test_split(data.drop(['Price'],axis=1), data["Price"], test_size = 0.30, random_state = 10)

# preskalujeme data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


def predict():

    df = pd.DataFrame(columns=['Model', 'Score', 'MSE'])

    # linearna regresia
    linear_reg_model = LinearRegression()
    linear_reg_model.fit(X_train, y_train)
    y_pred = linear_reg_model.predict(X_test)
    score = round(linear_reg_model.score(X_test, y_test), 2)
    MSE = round(metrics.mean_squared_error(y_test, y_pred), 2)
    df.loc[0] = ["Lineárna Regresia", score, MSE ]

    #ridge regresia
    ridge_reg_model = Ridge()
    ridge_reg_model.fit(X_train, y_train)
    y_pred = ridge_reg_model.predict(X_test)
    score = round(ridge_reg_model.score(X_test, y_test), 2)
    MSE = round(metrics.mean_squared_error(y_test, y_pred), 2)
    df.loc[1] = ["Ridge Regresia", score, MSE]

    # lasso regresia
    lasso_reg_model = linear_model.Lasso(alpha=0.1)
    lasso_reg_model.fit(X_train, y_train)
    y_pred = lasso_reg_model.predict(X_test)
    score = round(lasso_reg_model.score(X_test, y_test), 2)
    MSE = round(metrics.mean_squared_error(y_test, y_pred), 2)
    df.loc[2] = ["Lasso Regresia", score, MSE]

    # k neighbors regresia
    kneighbors_reg_model = KNeighborsRegressor(n_neighbors=5)
    kneighbors_reg_model.fit(X_train, y_train)
    y_pred = kneighbors_reg_model.predict(X_test)
    score = round(kneighbors_reg_model.score(X_test, y_test), 2)
    MSE = round(metrics.mean_squared_error(y_test, y_pred), 2)
    df.loc[3] = ["K Neighbors Regresia", score, MSE]

    # decision tree regresia
    # decisionTree_reg_model = DecisionTreeRegressor()
    # decisionTree_reg_model.fit(X_train, y_train)
    # y_pred = decisionTree_reg_model.predict(X_test)
    # print("DecisionTree Model")
    # print("Score:", round(decisionTree_reg_model.score(X_test, y_test), 2))
    # print("Mean Squared Error (MSE):", round(metrics.mean_squared_error(y_test, y_pred), 2))
    # print("-------")

    # baggging regresia
    bagging_reg_model = BaggingRegressor()
    bagging_reg_model.fit(X_train, y_train)
    y_pred = bagging_reg_model.predict(X_test)
    score = round(bagging_reg_model.score(X_test, y_test), 2)
    MSE = round(metrics.mean_squared_error(y_test, y_pred), 2)
    df.loc[4] = ["Bagging Regresia", score, MSE]

    # random forest regresia
    randomForest_reg_model = RandomForestRegressor()
    randomForest_reg_model.fit(X_train, y_train)
    y_pred = randomForest_reg_model.predict(X_test)
    score = round(randomForest_reg_model.score(X_test, y_test), 2)
    MSE = round(metrics.mean_squared_error(y_test, y_pred), 2)
    df.loc[5] = ["Random Forest Regresia", score, MSE]

    df.set_index("Model", inplace=True)
    print(df)

#predict()