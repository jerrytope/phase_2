import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score



def load_data():
    new_df = pd.read_csv('../data/predictions.csv')
    df2 = pd.read_csv('../data/relevance_data.csv')
    return new_df, df2

def merge_data(new_df, df2):
    merged_df = pd.merge(new_df, df2, on='DocumentID')
    return merged_df



def feature_engineering(merged_df):
    # Filter by relevant regulation
    filtered_df = merged_df[merged_df['ContainsRelevantRegulation'] == True]
    
    # Count DocumentIDs
    DocumentID_count = filtered_df['DocumentID'].value_counts().reset_index()
    DocumentID_count.columns = ['DocumentID', 'Count']
    
    # Count DocumentIDs in the entire dataset
    tope_3 = merged_df['DocumentID'].value_counts().reset_index()
    tope_3.columns = ['DocumentID', 'Count']
    
    # Merge counts and calculate percentage
    merged_DocumentID = pd.merge(DocumentID_count, tope_3, on='DocumentID', suffixes=('_document_counts', '_tope3'))
    merged_DocumentID['Percentage'] = (merged_DocumentID['Count_document_counts'] / merged_DocumentID['Count_tope3']) * 100
    
    # Assign points based on percentage
    def assign_points(percentage):
        if percentage < 33:
            return 1
        elif 33 <= percentage < 50:
            return 2
        elif 50 <= percentage < 75:
            return 3
        elif 75 <= percentage < 100:
            return 4
        elif percentage == 100:
            return 5
    
    merged_DocumentID['Points'] = merged_DocumentID['Percentage'].apply(assign_points)
    
    # Merge points back to the main DataFrame
    merged_df = merged_df.merge(merged_DocumentID[['DocumentID', 'Points']], on='DocumentID', how='left')
    merged_df['Points'].fillna(0, inplace=True)
    merged_df['Points'] = merged_df['Points'].astype(int)
    
    return merged_df




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
    
def clean_data(merged_df):
    merged_df['Cleaned_Content'] = merged_df['Content'].apply(clean_text)
    merged_df['Cleaned_Title'] = merged_df['Title'].apply(clean_text)
    return merged_df



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


def count_keywords(content, keywords):
    if isinstance(content, str):
        content_lower = content.lower()
        return sum(1 for keyword in keywords if keyword in content_lower)
    else:
        return 0

def add_keyword_features(merged_df):
    keywords_lower = [keyword.lower() for keyword in keywords]
    
    merged_df['keyword_count'] = merged_df['Cleaned_Content'].apply(lambda x: count_keywords(x, keywords_lower))
    merged_df['keywords_point'] = merged_df['keyword_count'] * 0.1
    
    merged_df['total_point'] = merged_df['Points'] + merged_df['keywords_point']
    merged_df['new_DocumentID'] = merged_df['total_point'].apply(lambda x: 1 if x > 4 else 0)
    merged_df['new_DocumentTypeId'] = merged_df['total_point'].apply(lambda x: 1 if x > 5 else 0)
    merged_df['new_RegulatorId'] = merged_df['total_point'].apply(lambda x: 1 if x > 4 else 0)
    merged_df['new_pdf'] = merged_df['IsPdf'].astype(int)
    
    return merged_df





def train_model(df):
    # Convert boolean target to binary
    df['ContainsRelevantRegulation'] = df['ContainsRelevantRegulation'].astype(int)
    
    # Text features
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    X_text = vectorizer.fit_transform(df['Cleaned_Content'] + ' ' + df['Cleaned_Title'])
    
    # Combine text features with other features
    X = pd.concat([pd.DataFrame(X_text.toarray()), df[['new_DocumentID', 'new_DocumentTypeId', 'new_RegulatorId', 'new_pdf']]], axis=1)
    y = df['ContainsRelevantRegulation']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    os.makedirs('../models', exist_ok=True)
    joblib.dump(model, '../models/regulation_predictor_model.pkl')
    
    # Make predictions and evaluate
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    
    return model


def run_pipeline():
    new_df, df2 = load_data()
    merged_df = merge_data(new_df, df2)
    merged_df = feature_engineering(merged_df)
    merged_df = clean_data(merged_df)
    merged_df = add_keyword_features(merged_df)
    model = train_model(merged_df)
    
run_pipeline()
