from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import os

app = FastAPI()

# Define the data model
class Document(BaseModel):
    DocumentID: str
    Title: str
    RegulatorId: str
    SourceLanguage: str
    DocumentTypeId: str
    PublicationDate: str
    IsPdf: bool
    Content: str

@app.post("/process_document/")
async def process_document(doc: Document):
    # Convert the input to a dictionary
    doc_dict = doc.dict()

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame([doc_dict])

    # Define the file path
    file_path = f"document_{doc.DocumentID}.csv"
    
    # Ensure the directory exists
    # os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Save the DataFrame to a CSV file
    df.to_csv(file_path, index=False)

    return {"message": "Document saved successfully", "file_path": file_path}

# Running the application with `uvicorn`:
# uvicorn main:app --reload
