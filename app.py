import numpy as np
import pandas as pd
import streamlit as st 
from sklearn import preprocessing
import pickle

model = pickle.load(open('model.pkl', 'rb'))
encoder_dict = pickle.load(open('encoder.pkl', 'rb')) 
cols=['age','workclass','education','marital-status','occupation','relationship','race','gender','capital-gain','capital-loss',
      'hours-per-week','native-country']    
  
def main(): 
    st.title("Income Predictor")
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Income Prediction App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    
    age = st.text_input("Age","0") 
    workclass = st.selectbox("Working Class", ["Federal-gov","Local-gov","Never-worked","Private","Self-emp-inc","Self-emp-not-inc","State-gov","Without-pay"]) 
    education = st.selectbox("Education",["10th","11th","12th","1st-4th","5th-6th","7th-8th","9th","Assoc-acdm","Assoc-voc","Bachelors","Doctorate","HS-grad","Masters","Preschool","Prof-school","Some-college"]) 
    marital_status = st.selectbox("Marital Status",["Divorced","Married-AF-spouse","Married-civ-spouse","Married-spouse-absent","Never-married","Separated","Widowed"]) 
    occupation = st.selectbox("Occupation",["Adm-clerical","Armed-Forces","Craft-repair","Exec-managerial","Farming-fishing","Handlers-cleaners","Machine-op-inspct","Other-service","Priv-house-serv","Prof-specialty","Protective-serv","Sales","Tech-support","Transport-moving"]) 
    relationship = st.selectbox("Relationship",["Husband","Not-in-family","Other-relative","Own-child","Unmarried","Wife"]) 
    race = st.selectbox("Race",["Amer Indian Eskimo","Asian Pac Islander","Black","Other","White"]) 
    gender = st.selectbox("Gender",["Female","Male"]) 
    capital_gain = st.text_input("Capital Gain","0") 
    capital_loss = st.text_input("Capital Loss","0") 
    hours_per_week = st.text_input("Hours per week","0") 
    nativecountry = st.selectbox("Native Country",["Cambodia","Canada","China","Columbia","Cuba","Dominican Republic","Ecuador","El Salvadorr","England","France","Germany","Greece","Guatemala","Haiti","Netherlands","Honduras","HongKong","Hungary","India","Iran","Ireland","Italy","Jamaica","Japan","Laos","Mexico","Nicaragua","Outlying-US(Guam-USVI-etc)","Peru","Philippines","Poland","Portugal","Puerto-Rico","Scotland","South","Taiwan","Thailand","Trinadad&Tobago","United States","Vietnam","Yugoslavia"]) 
    
    if st.button("Predict"): 
        features = [[age,workclass,education,marital_status,occupation,relationship,race,gender,capital_gain,capital_loss,hours_per_week,nativecountry]]
        data = {'age': int(age), 'workclass': workclass, 'education': education, 'maritalstatus': marital_status, 'occupation': occupation, 'relationship': relationship, 'race': race, 'gender': gender, 'capitalgain': int(capital_gain), 'capitalloss': int(capital_loss), 'hoursperweek': int(hours_per_week), 'nativecountry': nativecountry}
        print(data)
        df=pd.DataFrame([list(data.values())], columns=['age','workclass','education','maritalstatus','occupation','relationship','race','gender','capitalgain','capitalloss','hoursperweek','nativecountry'])
                
        category_col =['workclass', 'education', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
        for cat in encoder_dict:
            for col in df.columns:
                le = preprocessing.LabelEncoder()
                if cat == col:
                    le.classes_ = encoder_dict[cat]
                    for unique_item in df[col].unique():
                        if unique_item not in le.classes_:
                            df[col] = ['Unknown' if x == unique_item else x for x in df[col]]
                    df[col] = le.transform(df[col])
            
        features_list = df.values.tolist()      
        prediction = model.predict(features_list)
    
        output = int(prediction[0])
        if output == 1:
            text = ">50K"
        else:
            text = "<=50K"

        st.success('Employee Income is {}'.format(text))
      
if __name__=='__main__': 
    main() 
    