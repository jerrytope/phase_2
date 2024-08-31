FastAPI Document Relevance Prediction
Overview

This FastAPI application is designed to predict the relevance of documents based on their content and metadata. The application utilizes a pre-trained machine learning model to classify documents as either "Relevant" or "Irrelevant." Additionally, the confidence level of the prediction is provided.
Features

    Predict Document Relevance: Given the document's metadata and content, the API predicts whether the document is relevant or not.
    Confidence Score: The API returns the confidence level of the prediction.
    Keyword Analysis: The application performs keyword analysis on the document content to assist in feature engineering.

Installation and Setup
Install Dependencies

Make sure you have Python installed. Then, install the required dependencies using:

bash

pip install -r requirements.txt

Running the Application Locally

    Start the FastAPI Application:

    To start the FastAPI application, run:

    bash

uvicorn main:app --reload

Note: If you're working locally, make sure to uncomment the code on line 6 in app.py before running the application.

Start the Streamlit Application:

To start the Streamlit application, run:

bash

    streamlit run app.py

API Hosting

The API is hosted on AWS NGINX.

    FastAPI Service URL: http://44.204.152.90:8000/docs
    FastAPI Predict Endpoint: http://44.204.152.90:8000/predict/
    Streamlit Application: https://phase2.streamlit.app/

API Endpoints
1. Home Route

    Endpoint: /home/
    Method: GET
    Description: Simple endpoint to verify that the application is running.
    Response:
        Status code: 200 OK
        Message indicating that the server is running.

2. Predict Relevance

    Endpoint: /predict/
    Method: POST
    Description: Predicts the relevance of the provided document.
    Request Body:
        JSON object containing the document data as described in the Input Data section.
    Response:

    json

    {
      "DocumentID": "6dc0-4e62-496e-b788-55816f3b",
      "Relevance": "Relevant",
      "Confidence": 0.89
    }

Input and Output Data Formats
Input Data:

json

{
  "DocumentID": "string",
  "Title": "string",
  "RegulatorId": "string",
  "SourceLanguage": "string",
  "DocumentTypeId": "string",
  "PublicationDate": "string",
  "IsPdf": true,
  "Content": "string"
}

Output Data:

json

{
  "DocumentID": "string",
  "Relevance": "Relevant",
  "Confidence": 0.89
}

Usage

To predict the relevance of a document, send a POST request to the /predict/ endpoint with the appropriate JSON object in the request body. The response will include the predicted relevance and the confidence level.
