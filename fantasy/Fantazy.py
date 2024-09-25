import csv
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas
from sklearn.metrics import accuracy_score

    identity=[]
    evidence=[]
    labels=[]
    model = LinearRegression()
    i=0
    df = pandas.read_csv('merged_gw_2023.csv')
    last_row_index = df.index[-1]

    columns_to_include1=['name'	, 'position'	,'team']
    max_rows=index[i+1]
    max_rows1 = index[i + 2]
    df_2024 = pd.read_csv('gw1.csv')

    24_names = df_2024(['names'])
    24_names=list(set(my_list))
while i<last_row_index
    while i<38 :
            text_value = df.iat[i + 2, 1]
            if text_value in 24_names:

                csv=pandas.read_csv('merged_gw_2023.csv',nrows=i+1)
                identity_df=pandas.read_csv('merged_gw_2023.csv', usecols=columns_to_include1, nrows=i+1)
                labels_df=pandas.read_csv('merged_gw_2023.csv', usecols=['total_points'], nrows=i+1)
                evidence_df =csv.drop(columns=['total_points','name', 'position','team'])


                evidence = evidence_df.values.tolist()
                labels = labels_df.values.tolist()
                X_train, y_train = evidence, labels  # From the first few gameweeks of 2023
                model.fit(X_train, y_train)
            else :
                i=i+39
            i+=1

df_2024 = pd.read_csv('gw1.csv')
evidence_2024 = df_2024.drop(columns=['total_points', 'name', 'position', 'team']).values
labels_2024 = df_2024['total_points'].values
model.fit(evidence_2024, labels_2024)
predictions = model.predict(evidence_2024)
evidence_df2 = csv1.drop(columns=['total_points', 'name', 'position', 'team'])
def best_player(model,file2):
    user_prediction = model.predict([user_data])