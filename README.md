Overview

This FastAPI application is designed to predict the relevance of documents based on their content and metadata. The application uses a pre-trained machine learning model to classify documents as either "Relevant" or "Irrelevant." Additionally, the confidence level of the prediction is provided.
Features

    Predict Document Relevance: Given the document's metadata and content, the API predicts whether the document is relevant or not.
    Confidence Score: The API returns the confidence level of the prediction.
    Keyword Analysis: The application performs keyword analysis on the document content to assist in feature engineering.

RUN uvicorn main:app --reload to load the fastapi application 

# INPUT DATA
{
	"DocumentID": str
  "Title": str,
  "RegulatorId": str,
  "SourceLanguage": str,
  "DocumentTypeId": str,
  "PublicationDate": str,
  "IsPdf": bool,
  "Content": str,
}

# OUTPUT:
{
	"DocumentID": str
  "Relevance": enum["Relevant", "Irrelevant"],
  "Confidence": float[0 <= x <= 1],
  
}


API Endpoints
1. Home Route

    Endpoint: /home/

    Method: GET

    Description: Simple endpoint to verify that the application is running.

    Response:


2.  Predict Relevance

    Endpoint: /predict/

    Method: POST

    Description: Predicts the relevance of the provided document.

    Request Body:
        JSON object containing the document data as described in the Input Data section.

    Response:
                {
              "DocumentID": "6dc0-4e62-496e-b788-55816f3b",
              "Relevance": "Relevant",
              "Confidence": 0.89
            }

Install Dependencies:
pip install -r requirements.txt

Run the Application:
uvicorn main:app --reload
