"""
ClimaShield 2.0 - Data Model and ML Pipeline
Handles AIWP data loading, preprocessing, and AI risk prediction model.
"""

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


# Sample regions with coordinates (for demonstration)
# In production, this would come from the AIWP dataset
SAMPLE_REGIONS = [
    # High risk coastal regions (flood-prone)
    {"region": "Bangladesh - Dhaka", "lat": 23.8103, "lon": 90.4125, "country": "Bangladesh", "risk_bias": "high"},
    {"region": "Philippines - Manila", "lat": 14.5995, "lon": 120.9842, "country": "Philippines", "risk_bias": "high"},
    {"region": "Indonesia - Jakarta", "lat": -6.2088, "lon": 106.8456, "country": "Indonesia", "risk_bias": "high"},
    {"region": "Vietnam - Ho Chi Minh", "lat": 10.8231, "lon": 106.6297, "country": "Vietnam", "risk_bias": "high"},
    
    # Moderate risk regions
    {"region": "India - Mumbai", "lat": 19.0760, "lon": 72.8777, "country": "India", "risk_bias": "moderate"},
    {"region": "Pakistan - Karachi", "lat": 24.8607, "lon": 67.0011, "country": "Pakistan", "risk_bias": "moderate"},
    {"region": "Kenya - Nairobi", "lat": -1.2921, "lon": 36.8219, "country": "Kenya", "risk_bias": "moderate"},
    {"region": "Brazil - São Paulo", "lat": -23.5505, "lon": -46.6333, "country": "Brazil", "risk_bias": "moderate"},
    
    # Low risk regions (stable climate, better infrastructure)
    {"region": "Norway - Oslo", "lat": 59.9139, "lon": 10.7522, "country": "Norway", "risk_bias": "low"},
    {"region": "Canada - Toronto", "lat": 43.6532, "lon": -79.3832, "country": "Canada", "risk_bias": "low"},
    {"region": "Australia - Sydney", "lat": -33.8688, "lon": 151.2093, "country": "Australia", "risk_bias": "low"},
    {"region": "New Zealand - Auckland", "lat": -36.8485, "lon": 174.7633, "country": "New Zealand", "risk_bias": "low"},
    {"region": "Chile - Santiago", "lat": -33.4489, "lon": -70.6693, "country": "Chile", "risk_bias": "low"},
    
    # Mixed risk regions
    {"region": "Nigeria - Lagos", "lat": 6.5244, "lon": 3.3792, "country": "Nigeria", "risk_bias": "moderate"},
    {"region": "Mexico - Mexico City", "lat": 19.4326, "lon": -99.1332, "country": "Mexico", "risk_bias": "moderate"},
    {"region": "South Africa - Cape Town", "lat": -33.9249, "lon": 18.4241, "country": "South Africa", "risk_bias": "low"},
]


@st.cache_data
def load_aiwp_data(use_sample: bool = True) -> pd.DataFrame:
    """
    Load AIWP dataset from AWS S3 or generate sample data for demonstration.
    
    """
    if use_sample:
        # Generate synthetic climate data for demonstration
        # In production, replace with actual AIWP data loading
        np.random.seed(42)
        
        data = []
        for idx, region_info in enumerate(SAMPLE_REGIONS):
            # Set different random seed for each region for variety
            np.random.seed(42 + idx)
            
            # Get risk bias to generate appropriate feature values
            risk_bias = region_info.get("risk_bias", "moderate")
            
            # Generate climate features based on risk level
            if risk_bias == "high":
                # High risk: high precipitation, low elevation, high flood history
                precipitation_mm = np.random.uniform(200, 400)
                temp_anomaly_c = np.random.uniform(2, 5)
                soil_moisture = np.random.uniform(0.6, 0.9)
                vegetation_index = np.random.uniform(0.2, 0.5)
                flood_history_count = np.random.randint(5, 15)
                population_density = np.random.uniform(2000, 8000)
                elevation_m = np.random.uniform(0, 50)
                distance_to_water_km = np.random.uniform(0.1, 5)
            elif risk_bias == "low":
                # Low risk: moderate precipitation, higher elevation, low flood history
                precipitation_mm = np.random.uniform(50, 150)
                temp_anomaly_c = np.random.uniform(-1, 2)
                soil_moisture = np.random.uniform(0.3, 0.6)
                vegetation_index = np.random.uniform(0.6, 0.9)
                flood_history_count = np.random.randint(0, 3)
                population_density = np.random.uniform(500, 2000)
                elevation_m = np.random.uniform(50, 500)
                distance_to_water_km = np.random.uniform(10, 100)
            else:  # moderate
                # Moderate risk: balanced features
                precipitation_mm = np.random.uniform(100, 250)
                temp_anomaly_c = np.random.uniform(0, 3)
                soil_moisture = np.random.uniform(0.4, 0.7)
                vegetation_index = np.random.uniform(0.4, 0.7)
                flood_history_count = np.random.randint(2, 7)
                population_density = np.random.uniform(1000, 4000)
                elevation_m = np.random.uniform(20, 200)
                distance_to_water_km = np.random.uniform(2, 30)
            
            # Generate synthetic climate features
            # These would come from actual AIWP satellite/sensor data
            record = {
                "region": region_info["region"],
                "lat": region_info["lat"],
                "lon": region_info["lon"],
                "country": region_info["country"],
                # Climate features (synthetic, based on risk bias)
                "precipitation_mm": precipitation_mm,
                "temp_anomaly_c": temp_anomaly_c,
                "soil_moisture": soil_moisture,
                "vegetation_index": vegetation_index,
                "flood_history_count": flood_history_count,
                "population_density": population_density,
                "elevation_m": elevation_m,
                "distance_to_water_km": distance_to_water_km,
            }
            data.append(record)
        
        df = pd.DataFrame(data)
        return df
    
    else:
        # Load from AIWP AWS S3 bucket
        try:
            import boto3
            import io
            
            bucket = "ai-for-water-and-planet"
            key = "AIWP_Dataset_Sample.csv"
            
            s3 = boto3.client("s3", config=boto3.session.Config(signature_version='UNSIGNED'))
            obj = s3.get_object(Bucket=bucket, Key=key)
            df = pd.read_csv(io.BytesIO(obj["Body"].read()))
            
            # Sample to 500 rows for performance
            if len(df) > 500:
                df = df.sample(500).reset_index(drop=True)
            
            return df
            
        except Exception as e:
            st.warning(f"Could not load AIWP data from S3: {e}. Using sample data instead.")
            return load_aiwp_data(use_sample=True)


def preprocess_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """
    Preprocess and engineer features for ML model.
    
    Args:
        df: Raw DataFrame with climate data
    
    Returns:
        Tuple of (processed DataFrame, list of feature names)
    """
    df_processed = df.copy()
    
    # Select features for modeling
    feature_cols = [
        "precipitation_mm",
        "temp_anomaly_c",
        "soil_moisture",
        "vegetation_index",
        "flood_history_count",
        "population_density",
        "elevation_m",
        "distance_to_water_km"
    ]
    
    # Create composite risk indicators
    df_processed["precip_anomaly"] = (df_processed["precipitation_mm"] - df_processed["precipitation_mm"].mean()) / df_processed["precipitation_mm"].std()
    df_processed["low_elevation_risk"] = (df_processed["elevation_m"] < 100).astype(int)
    df_processed["high_density_risk"] = (df_processed["population_density"] > 2000).astype(int)
    
    # Add engineered features to feature list
    feature_cols.extend(["precip_anomaly", "low_elevation_risk", "high_density_risk"])
    
    return df_processed, feature_cols


def create_synthetic_risk_target(df: pd.DataFrame) -> np.ndarray:
    """
    Create synthetic risk target variable for model training.
    In production, this would be actual historical risk/damage data.
    
    Args:
        df: DataFrame with climate features
    
    Returns:
        Array of risk scores (0-1)
    """
    # Weighted combination of risk factors
    risk = (
        0.25 * (df["precipitation_mm"] / df["precipitation_mm"].max()) +
        0.20 * (df["temp_anomaly_c"] / df["temp_anomaly_c"].max()) +
        0.15 * (1 - df["soil_moisture"]) +
        0.15 * (1 - df["vegetation_index"]) +
        0.15 * (df["flood_history_count"] / df["flood_history_count"].max()) +
        0.10 * (df["population_density"] / df["population_density"].max())
    )
    
    # Add some noise to make it more realistic
    risk = risk + np.random.normal(0, 0.05, len(risk))
    
    # Normalize to 0-1 range
    risk = np.clip(risk, 0, 1)
    
    return risk


@st.cache_resource
def train_risk_model(df: pd.DataFrame, feature_cols: list) -> Tuple[RandomForestRegressor, StandardScaler]:
    """
    Train Random Forest model to predict climate displacement risk.
    
    Args:
        df: Preprocessed DataFrame
        feature_cols: List of feature column names
    
    Returns:
        Tuple of (trained model, fitted scaler)
    """
    # Create synthetic target variable
    y = create_synthetic_risk_target(df)
    
    # Prepare features
    X = df[feature_cols].copy()
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    print(f"Model Training Complete - Train R²: {train_score:.3f}, Test R²: {test_score:.3f}")
    
    return model, scaler


def predict_risk_scores(df: pd.DataFrame, model: RandomForestRegressor, scaler: StandardScaler, feature_cols: list) -> pd.DataFrame:
    """
    Generate risk predictions for all regions.
    
    Args:
        df: Preprocessed DataFrame
        model: Trained ML model
        scaler: Fitted scaler
        feature_cols: List of feature column names
    
    Returns:
        DataFrame with added predicted_risk_score column
    """
    df_with_predictions = df.copy()
    
    # Prepare features
    X = df_with_predictions[feature_cols].fillna(df_with_predictions[feature_cols].mean())
    X_scaled = scaler.transform(X)
    
    # Predict risk scores
    predictions = model.predict(X_scaled)
    
    # Ensure predictions are in valid range
    predictions = np.clip(predictions, 0, 1)
    
    df_with_predictions["predicted_risk_score"] = predictions
    
    return df_with_predictions


def get_feature_importance(model: RandomForestRegressor, feature_cols: list) -> pd.DataFrame:
    """
    Extract feature importance from trained model.
    
    Args:
        model: Trained Random Forest model
        feature_cols: List of feature names
    
    Returns:
        DataFrame with features and their importance scores
    """
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    return importance_df


def initialize_data_model() -> Tuple[pd.DataFrame, RandomForestRegressor, StandardScaler, list]:
    """
    Initialize the complete data model pipeline.
    Loads data, trains model, and generates predictions.
    
    Returns:
        Tuple of (predictions DataFrame, model, scaler, feature_cols)
    """
    # Load data
    df_raw = load_aiwp_data(use_sample=True)
    
    # Preprocess features
    df_processed, feature_cols = preprocess_features(df_raw)
    
    # Train model
    model, scaler = train_risk_model(df_processed, feature_cols)
    
    # Generate predictions
    df_predictions = predict_risk_scores(df_processed, model, scaler, feature_cols)
    
    return df_predictions, model, scaler, feature_cols


def get_region_data(df: pd.DataFrame, region_name: str) -> Optional[pd.Series]:
    """
    Get data for a specific region.
    
    Args:
        df: DataFrame with predictions
        region_name: Name of the region
    
    Returns:
        Series with region data or None if not found
    """
    region_data = df[df["region"] == region_name]
    if len(region_data) > 0:
        return region_data.iloc[0]
    return None


def calculate_base_rate_per_region(df: pd.DataFrame) -> dict:
    """
    Calculate AI-calibrated base rate per region based on historical patterns.
    
    Args:
        df: DataFrame with risk predictions
    
    Returns:
        Dictionary mapping region names to base rates
    """
    base_rates = {}
    
    for _, row in df.iterrows():
        region = row["region"]
        risk_score = row["predicted_risk_score"]
        
        # Base rate increases with regional risk
        # Range: $30-$80 per month
        base_rate = 30 + (risk_score * 50)
        base_rates[region] = round(base_rate, 2)
    
    return base_rates
