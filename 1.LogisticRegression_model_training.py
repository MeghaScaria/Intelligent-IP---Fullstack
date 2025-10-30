# train_model_v3_fixed.py - Modified for your original dataset structure
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import HashingVectorizer
import joblib
import ipaddress

print("Loading training data from your original dataset...")
df = pd.read_csv('new_indian_training_data.csv')

# Display dataset info to confirm structure
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Unique cities: {df['city'].nunique()}")

# --- Enhanced Feature Preparation ---
print("Preparing enhanced features for your dataset structure...")

def to_prefix24(ip: str) -> str:
    """Extract /24 prefix from IP"""
    if not isinstance(ip, str) or '.' not in ip:
        return 'prefix24_unknown'
    parts = ip.split('.')
    if len(parts) < 3:
        return 'prefix24_unknown'
    return f"prefix24_{parts[0]}.{parts[1]}.{parts[2]}"

def ip_to_numeric(ip: str) -> int:
    """Convert IP to numeric representation"""
    try:
        return int(ipaddress.ip_address(ip))
    except:
        return 0

def extract_cidr_length(cidr: str) -> int:
    """Extract CIDR length from CIDR notation"""
    if pd.isna(cidr) or not isinstance(cidr, str):
        return 24  # Default to /24
    try:
        if '/' in cidr:
            return int(cidr.split('/')[1])
        return 24
    except:
        return 24

def clean_rdns_tokens(rdns: str) -> str:
    """Clean and tokenize rDNS hostname"""
    if pd.isna(rdns) or rdns == 'no_rdns':
        return 'no_rdns'
    
    rdns = str(rdns).lower()
    
    # Extract meaningful tokens
    tokens = []
    
    # Domain parts
    parts = rdns.split('.')
    if len(parts) >= 2:
        tokens.append(f"domain_{parts[-2]}_{parts[-1]}")  # TLD + SLD
    
    # ISP indicators
    isp_keywords = ['broadband', 'dsl', 'cable', 'fiber', 'wireless', 'mobile', 'static', 'dynamic']
    for keyword in isp_keywords:
        if keyword in rdns:
            tokens.append(f"isp_{keyword}")
    
    # Geographic indicators
    geo_keywords = ['mumbai', 'delhi', 'bangalore', 'chennai', 'kolkata', 'pune', 'hyderabad']
    for keyword in geo_keywords:
        if keyword in rdns:
            tokens.append(f"geo_{keyword}")
    
    # Numeric patterns
    if any(char.isdigit() for char in rdns):
        tokens.append("has_numbers")
    
    return ' '.join(tokens) if tokens else 'no_rdns'

# Handle missing values in your dataset
print("Handling missing values...")
df['rdns_hostname'] = df['rdns_hostname'].fillna('no_rdns')
df['asn_description'] = df['asn_description'].fillna('Unknown')
df['asn'] = df['asn'].fillna('0')
df['lat'] = df['lat'].fillna(df['lat'].median())
df['lon'] = df['lon'].fillna(df['lon'].median())

# Add basic features using your dataset columns
df['prefix24_token'] = df['ip_start'].apply(to_prefix24)
df['ip_numeric'] = df['ip_start'].apply(ip_to_numeric)
df['cidr_length'] = df['cidr'].apply(extract_cidr_length)
df['rdns_cleaned'] = df['rdns_hostname'].apply(clean_rdns_tokens)

# --- Frequency Features ---
print("Computing frequency features...")

# Count samples per prefix24
prefix24_counts = df['prefix24_token'].value_counts()
df['prefix24_count'] = df['prefix24_token'].map(prefix24_counts)

# Count samples per ASN
asn_counts = df['asn'].value_counts()
df['asn_count'] = df['asn'].map(asn_counts)

# Stability: fraction of samples in each prefix24 assigned to the top city
prefix24_stability = {}
for prefix24 in df['prefix24_token'].unique():
    prefix_df = df[df['prefix24_token'] == prefix24]
    if len(prefix_df) > 1:
        top_city_count = prefix_df['city'].value_counts().iloc[0]
        stability = top_city_count / len(prefix_df)
        prefix24_stability[prefix24] = stability
    else:
        prefix24_stability[prefix24] = 1.0

df['prefix24_stability'] = df['prefix24_token'].map(prefix24_stability)

# --- Build Enhanced Text Features ---
def build_enhanced_text_row(row) -> str:
    """Build enhanced text features for your dataset"""
    tokens = []
    
    # ASN features from your dataset
    asn_desc = str(row.get('asn_description', '')).lower()
    asn_id = str(row.get('asn', ''))
    
    # Add ASN ID if available
    if asn_id not in (None, 'nan', '') and asn_id != '0':
        tokens.append(f"asn_{asn_id}")
    
    # Add ASN description tokens
    if asn_desc and asn_desc != 'unknown':
        # Split description into words and add as features
        desc_words = asn_desc.split()
        for word in desc_words[:3]:  # Take first 3 words
            if len(word) > 2:  # Only meaningful words
                tokens.append(f"asndesc_{word}")
    
    # Cleaned rDNS tokens
    rdns_tokens = str(row.get('rdns_cleaned', '')).strip()
    if rdns_tokens and rdns_tokens != 'no_rdns':
        tokens.append(rdns_tokens)
    
    # Prefix24
    p24 = str(row.get('prefix24_token', ''))
    if p24 and p24 != 'prefix24_unknown':
        tokens.append(p24)
    
    # CIDR length as categorical
    cidr_len = row.get('cidr_length', 24)
    if cidr_len <= 16:
        tokens.append("cidr_large")
    elif cidr_len <= 20:
        tokens.append("cidr_medium")
    elif cidr_len <= 24:
        tokens.append("cidr_small")
    else:
        tokens.append("cidr_tiny")
    
    # Frequency features as categorical
    prefix24_count = row.get('prefix24_count', 1)
    if prefix24_count >= 100:
        tokens.append("prefix24_very_common")
    elif prefix24_count >= 20:
        tokens.append("prefix24_common")
    elif prefix24_count >= 5:
        tokens.append("prefix24_uncommon")
    else:
        tokens.append("prefix24_rare")
    
    asn_count = row.get('asn_count', 1)
    if asn_count >= 1000:
        tokens.append("asn_very_large")
    elif asn_count >= 100:
        tokens.append("asn_large")
    elif asn_count >= 20:
        tokens.append("asn_medium")
    else:
        tokens.append("asn_small")
    
    # Stability features
    stability = row.get('prefix24_stability', 1.0)
    if stability >= 0.9:
        tokens.append("prefix24_very_stable")
    elif stability >= 0.7:
        tokens.append("prefix24_stable")
    elif stability >= 0.5:
        tokens.append("prefix24_unstable")
    else:
        tokens.append("prefix24_very_unstable")
    
    # Add state information from your dataset
    state = str(row.get('state', '')).lower()
    if state and state != 'unknown':
        tokens.append(f"state_{state}")
    
    # Add ISP information if available in your dataset
    isp = str(row.get('isp', '')).lower()
    if isp and isp != 'unknown' and isp != 'nan':
        tokens.append(f"isp_{isp}")
    
    return ' '.join(tokens).strip()

df['enhanced_text_features'] = df.apply(build_enhanced_text_row, axis=1)

# --- Feature Statistics ---
print("Feature statistics for your dataset:")
print(f"Total samples: {len(df)}")
print(f"Unique prefix24s: {df['prefix24_token'].nunique()}")
print(f"Unique ASNs: {df['asn'].nunique()}")
print(f"Unique cities: {df['city'].nunique()}")
print(f"Average prefix24_count: {df['prefix24_count'].mean():.1f}")
print(f"Average asn_count: {df['asn_count'].mean():.1f}")
print(f"Average prefix24_stability: {df['prefix24_stability'].mean():.3f}")

# HashingVectorizer for enhanced features
HASH_N_FEATURES = 2**18  # Large feature space for better accuracy
vectorizer = HashingVectorizer(
    n_features=HASH_N_FEATURES,
    analyzer='word',
    ngram_range=(1, 2),
    alternate_sign=False,
    norm='l2',
    lowercase=True
)

# Handle rare cities - remove cities with < 2 samples
city_counts = df['city'].value_counts()
keep_cities = city_counts[city_counts >= 2].index
filtered_df = df[df['city'].isin(keep_cities)].reset_index(drop=True)

print(f"After filtering rare cities: {len(filtered_df)} samples, {len(keep_cities)} cities")

# Transform features
X = vectorizer.transform(filtered_df['enhanced_text_features'])

# Encode target cities
city_encoder = LabelEncoder()
targets = city_encoder.fit_transform(filtered_df['city'])

# Split data
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, targets, test_size=0.2, random_state=42, stratify=targets
    )
    print("Using stratified train-test split")
except ValueError:
    print("Stratified split failed due to rare classes. Using non-stratified split.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, targets, test_size=0.2, random_state=42
    )

# --- Model Training ---
print("Training the LogisticRegression model...")
print(f"Training data shape: {X_train.shape}")
print(f"Number of classes: {len(np.unique(y_train))}")
print("Starting training...")

import time
start_time = time.time()

model = LogisticRegression(
    solver='liblinear',
    class_weight='balanced',
    max_iter=1000,
    tol=1e-4,
    C=1.0,
    random_state=42,
    n_jobs=-1
)

# Fit the model
model.fit(X_train, y_train)

training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds ({training_time/60:.1f} minutes)")

# --- Evaluation ---
print("Evaluating model accuracy...")
preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)
print(f"Model Accuracy on Test Data: {accuracy:.4f}")

# Show some predictions
print("\nSample predictions:")
unique_test_cities = np.unique(y_test)
for i in range(min(5, len(unique_test_cities))):
    city_idx = unique_test_cities[i]
    actual_city = city_encoder.inverse_transform([city_idx])[0]
    
    # Find a prediction for this city
    mask = y_test == city_idx
    if np.sum(mask) > 0:
        sample_idx = np.where(mask)[0][0]
        predicted_city = city_encoder.inverse_transform([preds[sample_idx]])[0]
        print(f"  {i+1}. Actual: {actual_city} -> Predicted: {predicted_city}")

# --- Save the model ---
print("Saving model and supporting files...")
joblib.dump(model, 'india_ip_city_model_fixed.joblib')
joblib.dump(city_encoder, 'india_city_label_encoder_fixed.joblib')
joblib.dump(vectorizer, 'india_vectorizer_fixed.joblib')

# Save feature configuration
feature_config = {
    'n_features': HASH_N_FEATURES,
    'ngram_range': (1, 2),
    'version': 'v3_fixed_for_original_dataset',
    'features_used': [
        'asn_id', 'asn_description', 'rdns_cleaned', 'prefix24_token',
        'cidr_length', 'prefix24_count', 'asn_count', 'prefix24_stability',
        'state', 'isp'
    ],
    'model_type': 'LogisticRegression',
    'solver': 'liblinear',
    'dataset_columns': df.columns.tolist()
}
joblib.dump(feature_config, 'india_model_config_fixed.joblib')

print("\n✅ Model training complete!")
print("📊 Final Results:")
print(f"   - Dataset: {len(df)} records, {df['city'].nunique()} cities")
print(f"   - After filtering: {len(filtered_df)} records, {len(keep_cities)} cities")
print(f"   - Test Accuracy: {accuracy:.4f}")
print(f"   - Training time: {training_time/60:.1f} minutes")
print(f"   - Model saved as: india_ip_city_model_fixed.joblib")