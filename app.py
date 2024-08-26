import streamlit as st
import pandas as pd
import numpy as np


# set the title 
st.title("Fake Job Classification")

# Load the dataset
df = pd.read_csv("fake_job_postings (1).csv")

# add the user imput fields
