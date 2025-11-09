"""
ClimaShield 2.0 - Utility Functions
Helper functions for data processing, risk calculations, and AI explainability.
"""

import numpy as np
import pandas as pd
from typing import Optional


def sigmoid(x: float, k: float = 1.0) -> float:
    """
    Apply sigmoid activation function to normalize risk scores.
    
    Args:
        x: Input value (risk score)
        k: Steepness parameter (default 1.0)
    
    Returns:
        Normalized value between 0 and 1
    """
    return 1 / (1 + np.exp(-k * x))


def normalize_feature(series: pd.Series, method: str = "minmax") -> pd.Series:
    """
    Normalize a pandas Series using specified method.
    
    Args:
        series: Input pandas Series
        method: Normalization method ('minmax' or 'zscore')
    
    Returns:
        Normalized pandas Series
    """
    if method == "minmax":
        return (series - series.min()) / (series.max() - series.min() + 1e-8)
    elif method == "zscore":
        return (series - series.mean()) / (series.std() + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def calculate_income_index(income: float, min_income: float = 100, max_income: float = 5000) -> float:
    """
    Calculate normalized income index (0 to 1).
    Lower income = higher index = higher subsidy.
    
    Args:
        income: User's monthly income in USD
        min_income: Minimum income threshold
        max_income: Maximum income threshold
    
    Returns:
        Income index between 0 and 1
    """
    clamped_income = np.clip(income, min_income, max_income)
    return (clamped_income - min_income) / (max_income - min_income)


def calculate_ai_premium(
    risk_score: float,
    income: float,
    base_rate: float = 50.0,
    risk_weight: float = 1.0,
    income_weight: float = 0.5
) -> dict:
    """
    Calculate AI-adjusted insurance premium based on risk score and income.
    
    Args:
        risk_score: Predicted risk score from ML model (0-1)
        income: User's monthly income in USD
        base_rate: Base premium rate in USD
        risk_weight: Weight for risk component
        income_weight: Weight for income subsidy component
    
    Returns:
        Dictionary with premium, payout, and breakdown details
    """
    # Calculate income index (higher = wealthier, lower subsidy)
    income_index = calculate_income_index(income)
    
    # AI-adjusted premium formula
    # premium = base_rate * sigmoid(risk_weight * risk_score + income_weight * (1 - income_index))
    combined_score = risk_weight * risk_score + income_weight * (1 - income_index)
    premium_multiplier = sigmoid(combined_score)
    
    monthly_premium = base_rate * premium_multiplier
    
    # Estimate payout based on risk level (higher risk = higher payout)
    # Typical insurance payout is 10-20x annual premium for disaster coverage
    annual_premium = monthly_premium * 12
    payout_multiplier = 10 + (risk_score * 10)  # 10x to 20x based on risk
    estimated_payout = annual_premium * payout_multiplier
    
    return {
        "monthly_premium": round(monthly_premium, 2),
        "annual_premium": round(annual_premium, 2),
        "estimated_payout": round(estimated_payout, 2),
        "risk_score": round(risk_score, 3),
        "income_index": round(income_index, 3),
        "premium_multiplier": round(premium_multiplier, 3),
        "subsidy_amount": round(base_rate - monthly_premium, 2) if monthly_premium < base_rate else 0
    }


def get_risk_category(risk_score: float) -> tuple:
    """
    Categorize risk score into human-readable categories.
    
    Args:
        risk_score: Predicted risk score (0-1)
    
    Returns:
        Tuple of (category_name, color_code)
    """
    if risk_score < 0.3:
        return ("Low Risk", "#2ecc71")  # Green
    elif risk_score < 0.5:
        return ("Moderate Risk", "#f39c12")  # Orange
    elif risk_score < 0.7:
        return ("High Risk", "#e67e22")  # Dark orange
    else:
        return ("Critical Risk", "#e74c3c")  # Red


def format_currency(amount: float, currency: str = "USD") -> str:
    """
    Format a number as currency string.
    
    Args:
        amount: Amount to format
        currency: Currency code (default USD)
    
    Returns:
        Formatted currency string
    """
    if currency == "USD":
        return f"${amount:,.2f}"
    else:
        return f"{amount:,.2f} {currency}"


def generate_risk_summary(risk_score: float, region_name: str) -> str:
    """
    Generate a human-readable risk summary for a region.
    
    Args:
        risk_score: Predicted risk score (0-1)
        region_name: Name of the region
    
    Returns:
        Risk summary text
    """
    category, _ = get_risk_category(risk_score)
    
    if risk_score < 0.3:
        description = "Climate conditions are relatively stable. Standard precautions recommended."
    elif risk_score < 0.5:
        description = "Some climate variability detected. Monitor weather updates regularly."
    elif risk_score < 0.7:
        description = "Elevated climate risk factors present. Consider evacuation planning."
    else:
        description = "Critical risk levels detected. Immediate action and insurance coverage recommended."
    
    return f"{region_name}: {category} ({description})"


def create_shap_explainer(model, X_train: pd.DataFrame):
    """
    Create a SHAP explainer for the trained model.
    
    Args:
        model: Trained scikit-learn model
        X_train: Training features DataFrame
    
    Returns:
        SHAP TreeExplainer object
    """
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        return explainer
    except ImportError:
        print("Warning: SHAP not installed. Install with: pip install shap")
        return None


def explain_prediction(explainer, X_sample: pd.DataFrame, feature_names: list) -> Optional[dict]:
    """
    Generate SHAP explanation for a single prediction.
    
    Args:
        explainer: SHAP explainer object
        X_sample: Single sample DataFrame (1 row)
        feature_names: List of feature names
    
    Returns:
        Dictionary with feature importance values
    """
    if explainer is None:
        return None
    
    try:
        import shap
        shap_values = explainer.shap_values(X_sample)
        
        # Get the SHAP values for the sample
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        # Create feature importance dictionary
        importance = {}
        for i, feature in enumerate(feature_names):
            importance[feature] = float(shap_values[0][i])
        
        # Sort by absolute importance
        importance = dict(sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True))
        
        return importance
    except Exception as e:
        print(f"Error generating SHAP explanation: {e}")
        return None


def check_alert_threshold(risk_score: float, threshold: float = 0.6) -> bool:
    """
    Check if risk score exceeds alert threshold.
    
    Args:
        risk_score: Predicted risk score (0-1)
        threshold: Alert threshold (default 0.6)
    
    Returns:
        True if alert should be triggered
    """
    return risk_score >= threshold


def generate_alert_message(risk_score: float, region_name: str) -> str:
    """
    Generate alert message based on risk level.
    
    Args:
        risk_score: Predicted risk score (0-1)
        region_name: Name of the region
    
    Returns:
        Alert message string
    """
    category, _ = get_risk_category(risk_score)
    
    if risk_score >= 0.7:
        return f"⚠️ CRITICAL ALERT: {category} detected in {region_name}. " \
               f"Immediate evacuation planning recommended. Explore insurance options."
    else:
        return f"⚠️ High flood risk detected in {region_name}. " \
               f"Consider securing insurance coverage and monitoring weather updates."
