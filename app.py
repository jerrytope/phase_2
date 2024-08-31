import streamlit as st
import requests
import json

# FastAPI URL - ensure this matches where your FastAPI is running
# FASTAPI_URL = "http://127.0.0.1:8000/predict/"
FASTAPI_URL = "http://44.204.152.90:8000/predict/"

# Streamlit interface
st.title("Document Relevance Predictor")

# Input fields
DocumentID = st.text_input("Document ID")
Title = st.text_input("Title")
RegulatorId = st.text_input("Regulator ID")
SourceLanguage = st.text_input("Source Language")
DocumentTypeId = st.text_input("Document Type ID")
PublicationDate = st.date_input("Publication Date")
IsPdf = st.checkbox("Is PDF")
Content = st.text_area("Content")

# Create a button to send the data
if st.button("Predict Relevance"):
    # Prepare the payload
    data = {
        "DocumentID": DocumentID,
        "Title": Title,
        "RegulatorId": RegulatorId,
        "SourceLanguage": SourceLanguage,
        "DocumentTypeId": DocumentTypeId,
        "PublicationDate": PublicationDate.strftime("%Y-%m-%d"),
        "IsPdf": IsPdf,
        "Content": Content
    }

    # Send POST request to the FastAPI endpoint
    try:
        response = requests.post(FASTAPI_URL, json=data)
        
        if response.status_code == 200:
            result = response.json()
            
            st.success("Prediction successful!")
            st.write(f"Document ID: {result['DocumentID']}")
            st.write(f"Relevance: {result['Relevance']}")
            st.write(f"Confidence: {result['Confidence']:.2f}")
        else:
            st.error(f"Error: {response.status_code}")
            st.json(response.json())  # Display error details if available
            
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")

