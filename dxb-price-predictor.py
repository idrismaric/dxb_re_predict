import subprocess

# Install required packages
subprocess.check_call(["pip", "install", "-r", "requirements.txt"])

import streamlit as st
import pandas as pd
import xgboost as xgb

model = xgb.Booster()
model.load_model("xgbmodelBus.xgb")
input_data = pd.read_excel('inputdata.xlsx')
training_data = pd.read_csv('ml_inputBus.csv')


st.write(""" 
# Price prediction for the Property in Business Bay

""")

styled_table = (
    f'<style>'
    f'table {{ width: 300px; margin-left: -20px; padding-left: 0; }}'  # Set the desired width and reduce the left margin and padding
    f'table th, table td {{ word-wrap: break-word; }}'  # Enable word wrapping
    f'</style>'
    f'{input_data.to_html()}'
)
st.markdown(styled_table, unsafe_allow_html=True)

dict = input_data.to_dict(orient='list')
st.write("input data converted to dictionary")
new_dict = {key: value[0] for key, value in dict.items()}
st.write("defined key and values")
mod_dict = {f'{key}_{value}' if isinstance(value, str) else key: (value if isinstance(value, (int, float)) else 1) for key, value in new_dict.items()}
mod_data_df = pd.DataFrame.from_dict(mod_dict, orient='index').T
st.write("replaced key with new values and replaced string values with 1. then, converted the mod dict to df")
training_columns = training_data.columns
st.write("training data columns taken")
new_data_df = pd.DataFrame(columns=training_columns)
st.write("formed an empty dataframe with training data columns")
new_data_df = pd.concat([new_data_df, mod_data_df], ignore_index=True, axis=0, sort=False)
st.write("append with modified dataframe")
new_data_df = new_data_df.fillna(0)
st.write("Filled zeros on the empty cells")


ml_input = new_data_df.iloc[:,1:]
ml_input = ml_input.values

ml_input =  xgb.DMatrix(ml_input)

prediction = model.predict(ml_input)

st.write("The Predicted value of the Property:", prediction)
