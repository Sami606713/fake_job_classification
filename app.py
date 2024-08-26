import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl

def load_model():
    with open("final_model.pkl", "rb") as f:
        model = pkl.load(f)
    return model

# set the title 
st.title("Fake Job Classification")

# Load the dataset
df = pd.read_csv("fake_job_postings (1).csv")
# st.dataframe(df.head())
# st.write(df['title'].unique())
# add the user imput fields
# title
title = st.selectbox("Title", df['title'].unique())
# location
location = st.selectbox("location", df['location'].unique())
# 'department'
department = st.selectbox("department", df['department'].unique())
# employment_type
employment_type = st.selectbox("employment_type", df['employment_type'].unique())

# required_experience
required_experience=st.selectbox("required_experience", df['required_experience'].unique())
# required_education
required_education=st.selectbox("required_education", df['required_education'].dropna().unique())

# industry
industry=st.selectbox("industry", df['industry'].dropna().unique())
# function
function=st.selectbox("function", df['function'].unique())

# salary_range
salary_range = st.text_area("salary_range")
# company_profile
company_profile = st.text_area("company_profile")
# Description
description = st.text_area("description")

# requirements
requirements = st.text_area("requirements")
# Benefits
benefits = st.text_area("benefits")

# Converting data into dictionary
dic={
    'title':title,
    'location':location,
    'department':department,
    'employment_type':employment_type,
    'required_experience':required_experience,
    'required_education':required_education,
    "salary_range":"",
    'industry':industry,
    'function':function,
    'company_profile':company_profile,
    'description':description,
    'requirements':requirements,
    'benefits':benefits
}
if st.button("Classify:"):
    st.write("Loading Model.....")
    model=load_model()
    df=pd.DataFrame(dic,index=[0])
    # add the text column
    df["clean_text"]= df['title'] + " "+ df['location'] + " "+ df['department'] + " "+ df['salary_range'] +  " " + df['company_profile'] + " " + df['description'] + " " + df['requirements'] + " " + df['benefits'] + " " + df['employment_type'] + " " + df['required_experience'] + " " + df['required_education'] + " " + df['industry'] + " " + df['function']
    
    # drop the columns
    df.drop(["title","location","department","salary_range","company_profile","description","requirements","benefits","employment_type","required_experience","required_education","industry","function"],axis=1,inplace=True)

    # make the prediction
    st.write("Making Prediction....")
    prediction=model.predict(df[['clean_text']]) 

    if prediction==0:
        st.write(df['clean_text'][0])
        st.success("Not A Fake Job")
    elif prediction==1:
        st.write(df['clean_text'][0])
        st.error("Fake Job")





