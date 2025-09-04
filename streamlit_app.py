# Humberside_crime_dashboard.py
# Fast Streamlit Predictive Analytics Dashboard for large crime dataset
# It is optimised to load and show results quickly on large datasets (~300k rows)
# It trains on a STRATIFIED SAMPLE and caches model to disk to avoid re-training on reloads
# It holds lightweight features & model for speedy inference

import os
import zipfile
import tempfile
from io import BytesIO

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ---------------------------
# Config - tune these for speed vs accuracy
# ---------------------------
SEED = 42
MAX_TRAIN_SAMPLES = 100000    # how many rows to use to train the model (stratified). ~40k is fast and representative
TFIDF_MAX_FEATURES = 1000    # reduce text vector size for speed
MODEL_PATH = "cached_crime_model.joblib"
ENC_PATH = "cached_encoders.joblib"

# ---------------------------
# Helpful utility functions
# ---------------------------
def detect_csv_in_zip(zip_path):
    """Return first csv file name inside zip, or None."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.lower().endswith(".csv"):
                return name
    return None

@st.cache_data(show_spinner=False)
def load_dataframe(path_or_zip="data/humberside_merged.zip"):
    """
    Load the CSV quickly and cache it.
    Accepts either:
      - a path to a ZIP containing a single CSV, or
      - a direct CSV path.
    This function only runs once per file-change thanks to st.cache_data.
    """
    if os.path.exists(path_or_zip) and path_or_zip.lower().endswith(".zip"):
        csv_name = detect_csv_in_zip(path_or_zip)
        if csv_name is None:
            raise FileNotFoundError("No CSV inside the ZIP.")
        with zipfile.ZipFile(path_or_zip, "r") as zf:
            with zf.open(csv_name) as f:
                df = pd.read_csv(f)
    elif os.path.exists(path_or_zip) and path_or_zip.lower().endswith(".csv"):
        df = pd.read_csv(path_or_zip)
    else:
        raise FileNotFoundError(f"Data file not found at {path_or_zip}")
    # Minimal cleaning useful for dashboard & modelling:
    df.columns = df.columns.str.strip()
    # ensuring important columns exist
    expected = ['Crime ID', 'Month', 'Reported by', 'Falls within', 'Longitude', 'Latitude',
                'Location', 'LSOA code', 'LSOA name', 'Crime type', 'Last outcome category']
    # keepping only columns that actually exist (robust)
    keep_cols = [c for c in expected if c in df.columns]
    df = df[keep_cols].copy()
    # small type fixes
    if 'Month' in df.columns:
        df['Month'] = pd.to_datetime(df['Month'], errors='coerce', format='%Y-%m')
    for c in ['Longitude', 'Latitude']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    # dropping rows that miss key info
    if 'Crime ID' in df.columns:
        df = df.dropna(subset=['Crime ID'])
    if 'Crime type' in df.columns:
        df = df.dropna(subset=['Crime type'])
    # dropping duplicate events by ID to avoid data bloat
    if 'Crime ID' in df.columns:
        df = df.drop_duplicates(subset=['Crime ID'])
    # small clean for 'Location' to be used in text features
    if 'Location' in df.columns:
        df['Location'] = df['Location'].astype(str).str.replace(r"On or near", "", case=False, regex=True).str.strip()
    return df

def prepare_modeling_frame(df, target_col='Crime type'):
    """
    Reduce cardinality of target (group tiny classes into 'Other') and return X,y.
    We intentionally keep only a handful of cheap features:
      - Location text (TF-IDF)
      - Latitude & Longitude (numeric)
      - Reported by (categorical label encoded)
      - Falls within (categorical label encoded)
    This small feature set keeps training fast.
    """
    df = df.copy()
    # Group very rare classes into 'Other' to keep the classifier small and balanced
    counts = df[target_col].value_counts()
    min_count = 200  # classes with fewer examples than this are grouped into 'Other'
    rare = counts[counts < min_count].index
    df[target_col] = df[target_col].apply(lambda x: 'Other' if x in rare else x)

    # Keeping only rows with coordinates for geospatial features (optional but helpful)
    if ('Longitude' in df.columns) and ('Latitude' in df.columns):
        df = df.dropna(subset=['Longitude', 'Latitude'])
    else:
        # if no coords, create dummy zeros
        df['Longitude'] = 0.0
        df['Latitude'] = 0.0

    # Keepping required columns - if missing, use safe defaults
    for col in ['Reported by', 'Falls within', 'Location']:
        if col not in df.columns:
            df[col] = 'Unknown'

    # return
    return df

# ---------------------------
# Model training / loading (cached)
# ---------------------------
@st.cache_resource(show_spinner=False)
def get_or_train_model(df,
                       target_col='Crime type',
                       sample_size=MAX_TRAIN_SAMPLES,
                       tfidf_features=TFIDF_MAX_FEATURES,
                       model_path=MODEL_PATH,
                       enc_path=ENC_PATH,
                       force_retrain=False):
    """
    Load a cached model from disk if exists; otherwise train on a stratified sample
    and save to disk. The @st.cache_resource decorator ensures the trained model
    is kept in memory during the session and not re-trained on every interaction.
    """
    # If model file exists and not forcing retrain, attempt to load
    if os.path.exists(model_path) and os.path.exists(enc_path) and not force_retrain:
        try:
            model_data = joblib.load(model_path)
            encoders = joblib.load(enc_path)
            # model_data expected: dict with keys 'pipeline' and 'target_le'
            return model_data['pipeline'], model_data['target_le'], encoders
        except Exception:
            # corruption or version mismatch -> retrain
            pass

    # preparing a modelling frame (group rare classes etc)
    df_prep = prepare_modeling_frame(df, target_col=target_col)

    # Keeping only relevant columns
    cols = ['Location', 'Longitude', 'Latitude', 'Reported by', 'Falls within', target_col]
    df_prep = df_prep[cols].dropna(subset=[target_col])

    # stratified sample on target to avoid class imbalance bias and keep representation
    if len(df_prep) > sample_size:
        # stratifying by target
        sample_frac = sample_size / len(df_prep)
        # using stratified sample by sampling within each class proportionally
        df_sample = (df_prep.groupby(target_col, group_keys=False)
                              .apply(lambda x: x.sample(frac=sample_frac, random_state=SEED)))
        # if groupby caused rounding and I got > sample_size, trim
        if len(df_sample) > sample_size:
            df_sample = df_sample.sample(n=sample_size, random_state=SEED)
    else:
        df_sample = df_prep.copy()

    # Preparing X and y
    X_text = df_sample['Location'].fillna("").astype(str).values
    X_coords = df_sample[['Longitude', 'Latitude']].fillna(0.0).values

    # Quick label encoders for small categoricals (fast)
    encoders = {}
    for col in ['Reported by', 'Falls within']:
        le = LabelEncoder()
        encoders[col] = le.fit(df_sample[col].astype(str).fillna('Unknown'))
    X_cat = np.column_stack([encoders[c].transform(df_sample[c].astype(str).fillna('Unknown')) for c in encoders])

    # Target encoder
    target_le = LabelEncoder()
    y = target_le.fit_transform(df_sample[target_col].astype(str))

    # TF-IDF (small vocabulary for speed)
    tfidf = TfidfVectorizer(max_features=tfidf_features, ngram_range=(1,2), min_df=3)
    X_tfidf = tfidf.fit_transform(X_text)  # sparse matrix

    # Merge features: I'll concatenate sparse tfidf + numeric + categorical (dense). This is To keep code simple,
    # I'll convert small numeric+cat to dense and hstack with sparse tfidf.
    from scipy.sparse import hstack
    # scaling coords quickly
    scaler = StandardScaler()
    X_coords_scaled = scaler.fit_transform(X_coords)
    # converting categorical ints to 2D array (float)
    X_cat_float = X_cat.astype(float)
    # Stacking all features: tfidf (sparse) + coords (dense) + cat (dense)
    X_final = hstack([X_tfidf, np.asarray(X_coords_scaled), np.asarray(X_cat_float)])

    # Choosing a fast classifier. LogisticRegression with 'saga' is usually fast for sparse data.
    # As a fallback using a small RandomForest (fewer trees).
    try:
        clf = LogisticRegression(max_iter=2000, solver='saga', multi_class='multinomial', random_state=SEED, n_jobs=-1)
        clf.fit(X_final, y)
    except Exception:
        clf = RandomForestClassifier(n_estimators=50, random_state=SEED, n_jobs=-1)
        clf.fit(X_final, y)

    # Saving everything into a pipeline-like dict for persistence
    pipeline = {
        'tfidf': tfidf,
        'scaler': scaler,
        'clf': clf,
        'features_meta': {
            'cat_cols': list(encoders.keys())
        }
    }

    # persisting to disk so future runs can reuse it
    joblib.dump({'pipeline': pipeline, 'target_le': target_le}, model_path, compress=3)
    joblib.dump(encoders, enc_path, compress=3)

    return pipeline, target_le, encoders

# ---------------------------
# Prediction helper
# ---------------------------
def predict_single(pipeline, encoders, target_le, location_text, lon, lat, reported_by, falls_within):
    """
    Preparing one row and predicting a class label quickly.
    """
    # transforming text
    X_tfidf = pipeline['tfidf'].transform([location_text])
    # coords scaler
    coords_scaled = pipeline['scaler'].transform([[lon, lat]])
    # encoding categorical cols using encoders (unknown labels mapped to most frequent via try/except)
    cat_vals = []
    for col, val in zip(['Reported by', 'Falls within'], [reported_by, falls_within]):
        le = encoders.get(col)
        if le is None:
            cat_vals.append(0.0)
        else:
            try:
                cat_vals.append(float(le.transform([str(val)])[0]))
            except Exception:
                # unseen label -> mapping to mode (0) or nearest
                cat_vals.append(0.0)
    # combining
    from scipy.sparse import hstack
    X_final = hstack([X_tfidf, np.asarray(coords_scaled), np.asarray([cat_vals])])
    pred_idx = pipeline['clf'].predict(X_final)[0]
    pred_label = target_le.inverse_transform([pred_idx])[0]
    return pred_label

# ---------------------------
# Streamlit UI (This is my UI)
# ---------------------------
st.set_page_config(layout="wide", page_title="Fast Crime Predictive Dashboard")
st.title("Fast Predictive Analytics Dashboard — Crime (Optimised for Large Data)")

# Sidebar controls
st.sidebar.header("Settings (performance)")
data_file = st.sidebar.text_input("Path to data (ZIP or CSV)", value="data/humberside_merged.zip")
sample_for_training = st.sidebar.slider("Max training samples (stratified)", min_value=2000, max_value=100000, value=MAX_TRAIN_SAMPLES, step=1000)
tfidf_feats = st.sidebar.slider("TF-IDF max features", 200, 5000, TFIDF_MAX_FEATURES, step=100)
force_retrain = st.sidebar.checkbox("Force retrain model now", value=False)

# Loading dataset (cached)
with st.spinner("Loading dataset..."):
    try:
        df = load_dataframe(data_file)
    except Exception as e:
        st.error(f"Unable to load data: {e}")
        st.stop()

st.sidebar.write(f"Rows loaded: {len(df):,}")

# Tabs layout
tab1, tab2, tab3 = st.tabs(["Overview", "EDA (fast)", "Prediction"])

# ----- Overview -----
with tab1:
    st.header("Dataset Overview")
    st.write("Quick summary (data is cached; visualisations use sampling to stay responsive).")
    st.write(df.head(5))
    st.write(df.describe(include='all'))

    # Time series mini-plot (sampled aggregation if months exist)
    if 'Month' in df.columns and df['Month'].notna().any():
        monthly = df.groupby(df['Month'].dt.to_period('M')).size().sort_index()
        # converting to numeric and show last 24 months
        monthly = monthly[-24:]
        st.line_chart(monthly.astype(int))

    # Top crime types (fast)
    if 'Crime type' in df.columns:
        top = df['Crime type'].value_counts().nlargest(15)
        fig, ax = plt.subplots()
        sns.barplot(x=top.values, y=top.index, ax=ax)
        ax.set_title("Top 15 Crime Types")
        st.pyplot(fig)

# ----- EDA (fast) and (swift) for good audience view-----
with tab2:
    st.header("Exploratory Data Analysis (fast & sample-based)")
    st.write("Visualisations below are created on a sampled subset for speed (adjust sample size if needed).")
    # sample for plots
    plot_sample_size = min(20000, len(df))
    df_sample = df.sample(n=plot_sample_size, random_state=SEED)

    # Map is built using Streamlit's native map (fast)
    if ('Latitude' in df.columns) and ('Longitude' in df.columns'):
        st.subheader("Geographic sample (fast map)")
        st.map(df_sample[['Latitude', 'Longitude']].rename(columns={'Latitude':'lat','Longitude':'lon'})[['lat','lon']].dropna())

    # Correlation heatmap for numerical features (very small)
    st.subheader("Numeric correlation (sample)")
    numeric_cols = [c for c in ['Longitude','Latitude'] if c in df_sample.columns]
    if numeric_cols:
        corr = df_sample[numeric_cols].corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, ax=ax)
        st.pyplot(fig)

    # Category frequency examples
    st.subheader("Category counts (fast)")
    for col in ['Falls within', 'Reported by', 'Last outcome category']:
        if col in df_sample.columns:
            topn = df_sample[col].value_counts().nlargest(10)
            fig, ax = plt.subplots()
            sns.barplot(x=topn.values, y=topn.index, ax=ax)
            ax.set_title(f"Top {len(topn)} {col}")
            st.pyplot(fig)

# ----- Prediction -----
with tab3:
    st.header("Crime Type Prediction (fast)")
    st.markdown("""
        This model is trained on a **stratified sample** (so minority classes are represented). 
        The sample size and TF-IDF size are configurable from the sidebar. After the first run,
        the trained model is cached to disk and reused to avoid re-training on each page refresh.
    """)

    # training / loading model (cached)
    with st.spinner("Preparing model (this may take ~30s on first run depending on sample size)..."):
        pipeline, target_le, encoders = get_or_train_model(df,
                                                          target_col='Crime type',
                                                          sample_size=sample_for_training,
                                                          tfidf_features=tfidf_feats,
                                                          force_retrain=force_retrain)

    st.success("Model ready — fast predictions available")

    # Simple input form for single prediction
    st.subheader("Single record prediction")
    with st.form("predict_form"):
        loc = st.text_input("Location (free text)", value="On or near Bellbrooke Avenue")
        lon = st.number_input("Longitude", value=float(df['Longitude'].median() if 'Longitude' in df.columns else 0.0))
        lat = st.number_input("Latitude", value=float(df['Latitude'].median() if 'Latitude' in df.columns else 0.0))
        rep = st.selectbox("Reported by", options=np.unique(df['Reported by'].fillna('Unknown').values)[:50])
        falls = st.selectbox("Falls within", options=np.unique(df['Falls within'].fillna('Unknown').values)[:50])
        submit = st.form_submit_button("Predict")

    if submit:
        pred = predict_single(pipeline, encoders, target_le, str(loc), float(lon), float(lat), rep, falls)
        st.info(f"Predicted Crime type: **{pred}**")

    # Batch prediction  (prediction on a sampled set and showing distribution)
    st.subheader("Batch predictions (sample)")
    run_batch = st.button("Run batch prediction on a small random sample (fast)")
    if run_batch:
        batch_size = min(10000, len(df))
        df_batch = df.sample(n=batch_size, random_state=SEED).copy()
        # Preparing arrays
        texts = df_batch['Location'].fillna("").astype(str).values
        coords = df_batch[['Longitude','Latitude']].fillna(0.0).values
        # Here transform
        X_tfidf = pipeline['tfidf'].transform(texts)
        coords_scaled = pipeline['scaler'].transform(coords)
        # This is my categorical encoding
        cat_array = []
        for c in ['Reported by','Falls within']:
            if c in df_batch.columns:
                le = encoders.get(c)
                if le is not None:
                    # unseen values cause transform to fail; Then fallback to 0
                    try:
                        cat_vals = le.transform(df_batch[c].astype(str).fillna('Unknown'))
                    except Exception:
                        cat_vals = np.zeros(len(df_batch), dtype=float)
                else:
                    cat_vals = np.zeros(len(df_batch), dtype=float)
            else:
                cat_vals = np.zeros(len(df_batch), dtype=float)
            cat_array.append(cat_vals.astype(float))
        cat_stack = np.column_stack(cat_array)
        
        # stacking
        from scipy.sparse import hstack
        X_final = hstack([X_tfidf, coords_scaled, cat_stack])
        preds_idx = pipeline['clf'].predict(X_final)
        preds = target_le.inverse_transform(preds_idx)
        df_batch['predicted_crime_type'] = preds
        st.write("Predicted distribution (sample):")
        st.write(df_batch['predicted_crime_type'].value_counts().head(20))
        fig, ax = plt.subplots(figsize=(8,6))
        sns.countplot(y=df_batch['predicted_crime_type'], order=df_batch['predicted_crime_type'].value_counts().index, ax=ax)
        ax.set_title("Predicted Crime Types (sample)")
        st.pyplot(fig)

# Footer: downloading small cleaned sample for reporting
st.sidebar.header("Quick sample download")
sample_dl = df.sample(n=min(2000, len(df)), random_state=SEED)
st.sidebar.download_button("Download sample CSV", data=sample_dl.to_csv(index=False).encode(), file_name="crime_sample.csv")

st.caption("Optimised dashboard: In this training uses stratified sampling + small TF-IDF + light classifier. "
           "For higher accuracy, I can increase sampling sizes & TF-IDF features in the sidebar (will cost more time).")
