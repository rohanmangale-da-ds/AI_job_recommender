import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Optional: stops oneDNN optimization logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # Suppresses TensorFlow INFO & WARNING logs

import streamlit as st
from src.recommender import recommend
from src.utils import format_jobs

st.title("💼 AI Job Recommendation System (Groq Gemini)")

st.write("Find jobs that match your skills and experience!")

# Collect user details
name = st.text_input("Enter your Name:")
job_preference = st.text_input("Enter your Job Preference:")
experience = st.text_area("Enter your Work Experience:")
skills = st.text_area("Enter your Skills:")
location_preference = st.text_input("Enter your Location Preference:")

if st.button("Get Recommendations"):
    # Combine all user inputs into a single profile string for recommendation
    user_profile = f"""
    Name: {name}
    Job Preference: {job_preference}
    Experience: {experience}
    Skills: {skills}
    Location Preference: {location_preference}
    """
    
    with st.spinner("Finding best matches..."):
        recommendations, similar_jobs = recommend(user_profile)
        
        st.subheader("🔍 Similar Jobs Found:")
        st.write(format_jobs(similar_jobs))
        
        st.subheader("🤖 AI Recommendations:")
        st.write(recommendations)


