import subprocess

# Install required packages
subprocess.check_call(["pip", "install", "-r", "requirements.txt"])

# Upgrade scipy using pip
subprocess.check_call(['pip', 'install', '--upgrade', 'scipy', '--user'])

import pandas as pd
import streamlit as st
from datasets import load_dataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb


# Load data and initialize variables
data = load_dataset('csv', data_files='msjdata/transactions/Transactions.csv')


# Define sidebar and filter data based on master project
st.sidebar.title('Price Prediction')
input_master = data['master_project_en'].dropna().unique() 
input_master = st.sidebar.selectbox('Select Master Project', input_master)
data = data[data['master_project_en'] == input_master]

#assigning input variables
procedure = data['procedure_name_en'].dropna().unique()
prop_type = data['property_type_en'].dropna().unique()
reg_type = data['reg_type_en'].dropna().unique()
area = data['area_name_en'].dropna().unique()
project = data['project_name_en'].dropna().unique()
metro = data['nearest_metro_en'].dropna().unique()
mall = data['nearest_mall_en'].dropna().unique()
rooms = data['rooms_en'].dropna().unique()


form_submitted = False
with st.form(key='my_form'):
        
        
        sqft = st.sidebar.number_input('Enter Square feet Area of Property')

        parking = st.sidebar.slider('Has Parking No/Yes(0/1)', 0, 1, 0)
                    
        procedure = st.sidebar.selectbox('Procedure name', procedure, index=0)

        prop_type = st.sidebar.selectbox('Property Type', prop_type, index=0)
        
        reg_type = st.sidebar.selectbox('Registration Type', reg_type, index=0)

        area = st.sidebar.selectbox('Area Name', area, index=0)

        project = st.sidebar.selectbox('Project Name', project, index=0)

        metro = st.sidebar.selectbox('Nearest Metro', metro, index=0)

        mall = st.sidebar.selectbox('Nearest Mall', mall, index=0)

        rooms = st.sidebar.selectbox('No. of Rooms', rooms, index=0)

        submit_button = st.form_submit_button(label='Process')

        if submit_button:
            form_submitted = True
            #st.experimental_rerun()

            if form_submitted:
                dict = {'procedure_name_en': procedure, 'property_type_en' : prop_type, 
            'reg_type_en' : reg_type, 'area_name_en' : area, 'master_project_en': input_master, 
            'project_name_en': project, 'nearest_metro_en' : metro, 'nearest_mall_en' : mall,
            'rooms_en' : rooms, 'has_parking' : parking, 'square_feet_area' : sqft}
        
                # Create square feet area column for filtered data
                data['square_feet_area'] = data['procedure_area'] * 10.77

                # Extract month from the date column
                data['year'] = pd.to_datetime(data['instance_date'], format='%d-%m-%Y').dt.year

                #Calculate sqft price
                data['sqft_price'] = data['actual_worth'] / data['square_feet_area']
                avg_price = data.groupby('year')['sqft_price'].mean()

                import matplotlib.pyplot as plt
                import numpy as np

                x = np.array(avg_price.index)
                y = np.array(avg_price.values)
                
                # Create the graph
                plt.plot(x, y)
                plt.xlabel('Date')
                plt.ylabel('Average Square Feet Price')
                plt.title('Average Square Feet Price Over Time')

                # Save the graph
                plt.savefig('average_price_graph.png')


                # Select relevant columns for machine learning
                data = data[['procedure_name_en', 'property_type_en', 'reg_type_en', 'area_name_en', 'master_project_en',
                    'project_name_en', 'nearest_metro_en', 'nearest_mall_en', 'rooms_en', 'has_parking',
                    'square_feet_area', 'actual_worth']]

                # Handle missing values
                categorical_columns = data.select_dtypes(include=['object']).columns
                data[categorical_columns] = data[categorical_columns].fillna(data[categorical_columns].mode().iloc[0])
                data = data.fillna(data.mode())

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
                st.write('RMSE:', int(rmse) )
                r2 = r2_score(y_test, y_pred)
                r2_perc = r2*100
                training_data = x.head(1)

                st.write('Regressor Score is =', round(r2,2), 'which means we have', round(r2_perc,2), 'percentage data to predict the price of the property in this area.' )


                st.write(""" 
                        # Price prediction for the Property in """, input_master)

        
                mod_dict = {f'{key}_{value}' if isinstance(value, str) else key: (value if isinstance(value, (int, float)) else 1) for key, value in dict.items()}
                mod_data_df = pd.DataFrame.from_dict(mod_dict, orient='index').T
                training_columns = training_data.columns
                new_data_df = pd.DataFrame(columns=training_columns)
                new_data_df = pd.concat([new_data_df, mod_data_df], ignore_index=True, axis=0, sort=False)
                new_data_df = new_data_df.fillna(0)


                ml_input = new_data_df.values

                #ml_input =  xgb.DMatrix(ml_input)
    
                prediction = xgb_model.predict(ml_input)

                st.write("The Predicted value of the Property:", prediction)

                from PIL import Image

                # Load the saved graph image
                image = Image.open('average_price_graph.png')

                # Display the graph using Streamlit
                st.image(image, caption='Average Square Feet Price of Area Over Time')