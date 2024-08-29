import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import joblib


# Define a list of keywords (same as before)
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

def clean_text(text):
    if isinstance(text, str):
        soup = BeautifulSoup(text, "html.parser")
        cleaned_text = soup.get_text()
        stop_words = set(stopwords.words('english'))
        words = cleaned_text.split()
        filtered_text = ' '.join([word for word in words if word.lower() not in stop_words])
        filtered_text = re.sub(r'[^\w\s]', '', filtered_text)
        return filtered_text
    else:
        return ''

def feature_engineering_and_keyword_count(data):
    data['Cleaned_Content'] = clean_text(data['Content'])
    data['Cleaned_Title'] = clean_text(data['Title'])
    
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

def predict_relevance(data, model, vectorizer):
    X_text = vectorizer.transform([data['Cleaned_Content'] + ' ' + data['Cleaned_Title']])
    X = pd.concat([pd.DataFrame(X_text.toarray()), pd.DataFrame([[data['new_DocumentID'], data['new_DocumentTypeId'], data['new_RegulatorId'], data['new_pdf']]])], axis=1)
    
    prediction = model.predict(X)[0]
    confidence = max(model.predict_proba(X)[0])
    
    return prediction, confidence

def run_data_pipeline(input_data, model, vectorizer):
    processed_data = feature_engineering_and_keyword_count(input_data)
    prediction, confidence = predict_relevance(processed_data, model, vectorizer)
    
    # Map prediction to relevant or irrelevant
    relevance_status = "Relevant" if prediction == 1 else "Irrelevant"
    
    return {
        'DocumentID': input_data['DocumentID'],
        'Relevance': relevance_status,
        'Confidence': confidence
    }

# Example usage
input_data = {
    "DocumentID": "6dc0-4e62-496e-b788-55816f3b",
    "Title": "SATURDAY November 4 2023 Official Gazette Issue 32359 Presidency Topic Energy Savings Public Buildings GENERALIZATION",
    "RegulatorId": "25e0-8863-4764-a512-6676e4cb",
    "SourceLanguage": "English",
    "DocumentTypeId": "9d3a-714e-4ddf-b821-61fd69ea",
    "PublicationDate": "2023-12-04",
    "IsPdf": True,
    "Content": "PresidencySubject Energy Saving Public BuildingsCIRCULAR2023115In order use public resources efficiently reduce energy costsEnergy manager according Energy Efficiency Law No 5627 dated 18042007015 determined Circular No 201918 public buildings obliged appointThe energy saving target updated minimum 9630 2030Achieving energy efficiency emission reduction reaching determined savings targetPrepared coordination Ministry Energy Natural Resources order ensureSavings Target Implementation Guide Public Buildings 20242030 Official picture MinistryIt published websiteAchievement savings target question public institutions organizationsto follow applications report energy savings calculations Presidencynotification Ministry Energy Natural Resources accordance procedure specified aforementioned RebberI would like kindly request information instructions regarding continuationNovember 3 2023Recep Tayyip ErdoganPRESIDENT"
}


model = joblib.load('model/regulation_predictor_model.pkl')
vectorizer = joblib.load('model/tfidf_vectorizer.pkl')

output = run_data_pipeline(input_data, model, vectorizer)
print(output)
