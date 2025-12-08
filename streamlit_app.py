import streamlit as st
import pandas as pd
from src.mlproject.pipeliness.prediction_pipeline import PredictPipeline

st.title("Student Performance Prediction")

gender = st.selectbox("Gender", ["male", "female"])
race = st.selectbox(
    "Race/Ethnicity",
    ["group A","group B","group C","group D","group E"]
)

parental = st.selectbox(
    "Parental Level of Education",
    [
        "some high school","high school","some college",
        "associate's degree","bachelor's degree","master's degree"
    ]
)

lunch = st.selectbox("Lunch", ["standard", "free/reduced"])
test_prep = st.selectbox("Test Preparation", ["none", "completed"])

reading = st.number_input("Reading Score", 0, 100)
writing = st.number_input("Writing Score", 0, 100)

if st.button("Predict"):
    data = pd.DataFrame([{
        "gender": gender,
        "race_ethnicity": race,
        "parental_level_of_education": parental,
        "lunch": lunch,
        "test_preparation_course": test_prep,
        "reading_score": reading,
        "writing_score": writing
    }])

    pipeline = PredictPipeline()
    result = pipeline.predict(data)

    st.success(f"Predicted Math Score: {round(result[0], 2)}")