from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import re

app = FastAPI()

# Load the pre-trained model and vectorizer
model = joblib.load('model/regulation_predictor_model2.pkl')
vectorizer = joblib.load('model/tfidf_vectorizer.pkl')

class DocumentInput(BaseModel):
    DocumentID: str
    Title: str
    RegulatorId: str
    SourceLanguage: str
    DocumentTypeId: str
    PublicationDate: str
    IsPdf: bool
    Content: str

# Clean the text content
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return ' '.join(re.findall(r'\w+', text.lower()))

# Feature engineering and keyword counting
def feature_engineering_and_keyword_count(data):
    data['Cleaned_Content'] = clean_text(data['Content'])
    data['Cleaned_Title'] = clean_text(data['Title'])

    # Count keywords in the cleaned content
    def count_keywords(content):
        content_lower = content.lower()
        return sum(1 for keyword in keywords if keyword.lower() in content_lower)
    
    data['keyword_count'] = count_keywords(data['Cleaned_Content'])
    data['keywords_point'] = data['keyword_count'] * 0.1
    
    data['new_DocumentID'] = 1 if data['keywords_point'] > 4 else 0
    data['new_DocumentTypeId'] = 1 if data['keywords_point'] > 5 else 0
    data['new_RegulatorId'] = 1 if data['keywords_point'] > 4 else 0
    data['new_pdf'] = int(data['IsPdf'])
    
    return data


keywords = [
    'financial', 'information', 'bank', 'article', 'date', 
    'securities', 'republic', 'paragraph', 'credit', 'data', 
    'their', 'risk', 'section', 'services', 'legal', 
    'accordance', 'reporting', 'all', 'state', 'foreign', 
    'person', 'market', '$', 'following', 'payment', 
    'investment', 'business', 'form', 'within', 'kazakhstan', 
    'management', 'provided', 'act', 'amount', 'requirements', 
    'account', 'exchange', 'service', 'public', 'electronic', 
    'national', 'case', 'been', 'into', 'tax', 'regulation',
    'Compliance', 'Capital', 'Equity', 'Debt', 'Liability', 
    'Contract', 'Regulation', 'Jurisdiction', 'Governance', 
    'Fraud', 'Penalty', 'Transaction', 'Asset', 'Treasury', 
    'Audit', 'Disclosure', 'Insolvency', 'Bankruptcy', 
    'Merger', 'Acquisition', 'Divestiture', 'Antitrust', 
    'Fiduciary', 'Interest', 'Dividend', 'Bond', 'Stock', 
    'Shareholder', 'Portfolio', 'Arbitration', 'Litigation', 
    'Reconciliation', 'Custodian', 'Brokerage', 'Underwriting', 
    'Hedge', 'Derivative', 'Swap', 'Option', 'Valuation', 
    'Prospectus', 'Collateral', 'Leverage', 'Liquidation', 
    'Monetary', 'Remittance', 'Escrow', 'Fiscal'
]

# Model prediction
def predict_relevance(data, model, vectorizer):
    X_text = vectorizer.transform([data['Cleaned_Content'] + ' ' + data['Cleaned_Title']])
    
    X = pd.concat([pd.DataFrame(X_text.toarray()), 
                   pd.DataFrame([[data['new_DocumentID'], data['new_DocumentTypeId'], data['new_RegulatorId'], data['new_pdf']]])], axis=1)
    
    prediction = model.predict(X)[0]
    confidence = max(model.predict_proba(X)[0])
    
    return prediction, confidence

@app.get("/")
def home():
    return {
        "name": "Temitope",
        "message": "Welcome!"
    }

@app.post("/predict/")
def predict(document: DocumentInput):
    input_data = document.dict()
    processed_data = feature_engineering_and_keyword_count(input_data)
    prediction, confidence = predict_relevance(processed_data, model, vectorizer)

    relevance_status = "Relevant" if prediction == 1 else "Irrelevant"
    
    return {
        'DocumentID': input_data['DocumentID'],
        'Relevance': relevance_status,
        'Confidence': confidence
    }
