import subprocess

# Upgrade scipy using pip
subprocess.check_call(['pip', 'install', '--upgrade', 'scipy', '--user'])


def train_xgboost_model(master_project):
    

    import streamlit as st
    import xgboost as xgb 
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error
    # Load data
    data_load = pd.read_csv('D:/Dropbox/00 Documents/01 Documents/02 Work/03 Tableau\DLD Dashboards/Transactions.csv')

    # Filter data based on master_project_en input
    data = data_load[data_load['master_project_en'] == master_project]
    
    # Create square feet area column
    data['square_feet_area'] = data['procedure_area'] * 10.77

    # Select relevant columns for machine learning
    data = data[['procedure_name_en', 'property_type_en', 'reg_type_en', 'area_name_en', 'master_project_en',
                    'project_name_en', 'nearest_metro_en', 'nearest_mall_en', 'rooms_en', 'has_parking',
                    'square_feet_area', 'actual_worth']]

    # Handle missing values
    categorical_columns = data.select_dtypes(include=['object']).columns
    data[categorical_columns] = data[categorical_columns].fillna(data[categorical_columns].mode().iloc[0])
    data = data.fillna(data.median())

    # Convert categorical columns to one-hot encoded features
    cat_cols = data.select_dtypes(include=['object']).columns.tolist()
    data_ml = pd.get_dummies(data, columns=cat_cols)


    # Split the data into training and testing sets
    x = data_ml.drop('actual_worth', axis=1)
    X = x.values
    y = data_ml['actual_worth'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and fit the XGBoost model
    xgb_model = xgb.XGBRegressor()
    
    xgb_model.fit(X_train, y_train)
    

    # Make predictions on the test set
    y_pred = xgb_model.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    st.write('RMSE:', rmse )
    r2 = r2_score(y_test, y_pred)
    st.write('Regressor Score is =', r2)

    training_data = x.head(1)

    st.write(""" 
    # Price prediction for the Property in """, master_project)

    sqft = st.sidebar.slider('Square feet Area of Property', 0 , 50000, 1000)

    parking = st.sidebar.slider('Has Parking No/Yes(0/1)', 0, 1, 0)

    master = data['master_project_en'].unique()
    master = st.sidebar.selectbox('Master Project', master, index=1)

    procedure = data['procedure_name_en'].unique()
    procedure = st.sidebar.selectbox('Procedure name', procedure, index=1)

    prop_type = data['property_type_en'].unique()
    prop_type = st.sidebar.selectbox('Property Type', prop_type, index=1)

    reg_type = data['reg_type_en'].unique()
    reg_type = st.sidebar.selectbox('Registration Type', reg_type, index=1)

    area = data['area_name_en'].unique()
    area = st.sidebar.selectbox('Area Name', area, index=1)

    project = data['project_name_en'].unique()
    project = st.sidebar.selectbox('Project Name', project, index=1)

    metro = data['nearest_metro_en'].unique()
    metro = st.sidebar.selectbox('Nearest Metro', metro, index=1)

    mall = data['nearest_mall_en'].unique()
    mall = st.sidebar.selectbox('Nearest Mall', mall, index=1)

    rooms = data['rooms_en'].unique()
    rooms = st.sidebar.selectbox('No. of Rooms', rooms, index=1)

    dict = {'procedure_name_en': procedure, 'property_type_en' : prop_type, 
              'reg_type_en' : reg_type, 'area_name_en' : area, 'master_project_en': master, 
              'project_name_en': project, 'nearest_metro_en' : metro, 'nearest_mall_en' : mall,
              'rooms_en' : rooms, 'has_parking' : parking, 'square_feet_area' : sqft}


    mod_dict = {f'{key}_{value}' if isinstance(value, str) else key: (value if isinstance(value, (int, float)) else 1) for key, value in dict.items()}
    training_columns = training_data.columns
    new_data_df = pd.DataFrame(columns=training_columns)
    new_data_df = new_data_df.append(mod_dict, ignore_index = True)
    new_data_df = new_data_df.fillna(0)


    ml_input = new_data_df.values

    ml_input =  xgb.DMatrix(ml_input)

    prediction = xgb_model.predict(ml_input)

    st.write("The Predicted value of the Property:", prediction)

train_xgboost_model('800 Villas')

