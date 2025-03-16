import streamlit as st
import pandas as pd
import joblib


# Gender :                   Male
# Age :                      44
# Driving_License :          1
# Region_Code :              28
# Previously_Insured :       0
# Vehicle_Age :             >2 years
# Vehicle_Damage :          yes
# Annual_Premium :          40454
# Policy_Sales_Channel :    26
# Vintage                   217


st.title('Customer interest in Policy')

#read the dataset to fill the values in the drop down list
df = pd.read_csv('train.csv')

#create the input fields
df['Gender'] = df['Gender'].str.strip()
Gender = st.selectbox("Gender", pd.unique(df['Gender']))
Age = st.number_input("Age")
Driving_License = st.selectbox("Driving_License", pd.unique(df['Driving_License']))
Region_Code = st.selectbox("Region_Code", pd.unique(df['Region_Code']))
Previously_Insured = st.selectbox("Previously_Insured", pd.unique(df['Previously_Insured']))
Vehicle_Age = st.selectbox("Vehicle_Age", pd.unique(df['Vehicle_Age']))
Vehicle_Damage = st.selectbox("Vehicle_Damage", pd.unique(df['Vehicle_Damage']))
Annual_Premium = st.number_input("Annual_Premium")
Policy_Sales_Channel = st.selectbox("Policy_Sales_Channel", pd.unique(df['Policy_Sales_Channel']))
Vintage = st.selectbox("Vintage", pd.unique(df['Vintage']))


#convert the input values into a dictionary
inputs = {
"Gender": Gender,
"Age": Age,
"Driving_License": Driving_License,
"Region_Code": Region_Code,
"Previously_Insured": Previously_Insured,
"Vehicle_Age": Vehicle_Age,
"Vehicle_Damage": Vehicle_Damage,
"Annual_Premium": Annual_Premium,
"Policy_Sales_Channel": Policy_Sales_Channel,
"Vintage": Vintage
}

inputs.update({'Gender':{'Male':0,'Female':1},
            'Vehicle_Damage':{'No':0,'Yes':1},
            'Vehicle_Age':{'< 1 Year':0,'1-2 Year':1,'> 2 Years':3}}
        )


#Click for prediction (prediction button)

if st.button("Predict"):
    model = joblib.load("policybuy_pipeline_model.pkl",mmap_mode='r')
    #input to the fields
    X_input = pd.DataFrame(inputs, index=[0])
    #model prediction
    prediction = model.predict(X_input)
    #display the prediction
    st.write(prediction)
