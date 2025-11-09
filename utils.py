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


def get_relocation_opportunities(region_name: str, risk_score: float) -> list:
    """
    Generate relocation opportunities, grants, and resources for high/moderate risk regions.
    
    Args:
        region_name: Name of the region
        risk_score: Predicted risk score (0-1)
    
    Returns:
        List of opportunity dictionaries with title, description, type, and eligibility
    """
    # Only provide opportunities for moderate to high risk regions
    if risk_score < 0.3:
        return []
    
    # Region-specific opportunities (mock data)
    opportunities_map = {
        "Bangladesh - Dhaka": [
            {
                "title": "🏘️ Climate Resilient Housing Program",
                "description": "Government-backed initiative providing low-interest loans and grants up to $15,000 for relocating to elevated areas in Chittagong Hill Tracts or Sylhet.",
                "type": "Relocation Grant",
                "eligibility": "Families in flood-prone zones with income <$5000/year",
                "deadline": "Rolling applications"
            },
            {
                "title": "🌾 Rural Livelihood Transition Fund",
                "description": "IOM-supported program offering vocational training and $3,000 startup capital for climate migrants transitioning to sustainable agriculture in safer regions.",
                "type": "Economic Opportunity",
                "eligibility": "Age 18-55, willing to relocate within Bangladesh",
                "deadline": "June 2026"
            },
            {
                "title": "🏠 Safe Haven Relocation Assistance",
                "description": "UNHCR partnership providing temporary housing, transportation costs, and 6-month rent subsidy for families relocating to climate-resilient districts.",
                "type": "Relocation Support",
                "eligibility": "Verified residents of high-risk flood zones",
                "deadline": "Emergency program - Apply anytime"
            }
        ],
        "India - Mumbai": [
            {
                "title": "🏙️ Smart City Relocation Incentive",
                "description": "Maharashtra State program offering ₹500,000 ($6,000) relocation bonus plus priority housing allocation in Pune or Nashik smart city developments.",
                "type": "Relocation Grant",
                "eligibility": "Mumbai residents in coastal flood zones",
                "deadline": "March 2026"
            },
            {
                "title": "💼 Green Jobs Migration Program",
                "description": "Skill India initiative with renewable energy training and guaranteed job placement in inland cities. Includes relocation allowance of ₹200,000.",
                "type": "Economic Opportunity",
                "eligibility": "All ages, basic education required",
                "deadline": "Rolling admissions"
            },
            {
                "title": "🏡 Climate Refugee Resettlement Scheme",
                "description": "Central Government scheme providing subsidized housing in 50+ climate-safe cities with preference for Mumbai coastal residents.",
                "type": "Relocation Support",
                "eligibility": "Income <₹8 lakhs/year, proof of climate risk",
                "deadline": "Open enrollment"
            }
        ],
        "Philippines - Manila": [
            {
                "title": "🌴 Safe Island Relocation Program",
                "description": "DSWD program providing ₱300,000 ($5,400) grant and land allocation in Mindanao highlands for typhoon-affected Metro Manila families.",
                "type": "Relocation Grant",
                "eligibility": "Families in NDRRMC high-risk zones",
                "deadline": "December 2025"
            },
            {
                "title": "⚓ Livelihood Seeding Project",
                "description": "ADB-funded initiative offering fishing/agriculture training and ₱150,000 startup capital for coastal families relocating inland.",
                "type": "Economic Opportunity",
                "eligibility": "Coastal barangay residents",
                "deadline": "February 2026"
            },
            {
                "title": "🏘️ Emergency Housing Voucher Program",
                "description": "Immediate housing assistance covering 12-month rent in safer provinces (Baguio, Davao, Cebu) for disaster-displaced families.",
                "type": "Relocation Support",
                "eligibility": "Storm surge/flood victims with documentation",
                "deadline": "Active during typhoon season"
            }
        ],
        "Pakistan - Karachi": [
            {
                "title": "🏔️ Northern Highlands Resettlement",
                "description": "NDMA program providing 5 marla plots and Rs.1,000,000 ($3,600) construction grant in Abbottabad or Murree for coastal flood victims.",
                "type": "Relocation Grant",
                "eligibility": "Karachi coastal belt residents",
                "deadline": "June 2026"
            },
            {
                "title": "🛠️ Skill Migration Initiative",
                "description": "UNDP partnership offering technical training and Rs.500,000 relocation support for moving to Islamabad/Lahore economic zones.",
                "type": "Economic Opportunity",
                "eligibility": "Age 18-50, any skill level",
                "deadline": "Rolling program"
            },
            {
                "title": "🏠 Climate Displacement Assistance",
                "description": "Provincial government aid covering transportation and 6-month rental subsidy for families relocating to inland Sindh or Punjab.",
                "type": "Relocation Support",
                "eligibility": "Registered IDPs from flood zones",
                "deadline": "Emergency relief - ongoing"
            }
        ],
        "Indonesia - Jakarta": [
            {
                "title": "🏙️ New Capital City Migration Incentive",
                "description": "Government program offering Rp 100 million ($6,500) grant plus job priority for Jakarta residents relocating to Nusantara (new capital).",
                "type": "Relocation Grant",
                "eligibility": "Jakarta residents in sinking/flood zones",
                "deadline": "December 2025"
            },
            {
                "title": "🌾 Transmigration Program 2.0",
                "description": "Modern transmigrasi with Rp 50 million, land allocation, and livelihood training in Kalimantan or Sulawesi.",
                "type": "Economic Opportunity",
                "eligibility": "Families willing to relocate inter-island",
                "deadline": "Ongoing recruitment"
            },
            {
                "title": "🏘️ BNPB Emergency Relocation Fund",
                "description": "Disaster agency providing immediate housing assistance and Rp 25 million support for moving to safer districts in Java.",
                "type": "Relocation Support",
                "eligibility": "Verified flood victims",
                "deadline": "Available during flood events"
            }
        ],
        "Vietnam - Ho Chi Minh": [
            {
                "title": "🏞️ Central Highlands Resettlement",
                "description": "Ministry of Agriculture offering 500m² land plots and ₫200 million ($8,000) for Mekong Delta families relocating to Da Lat or Kon Tum.",
                "type": "Relocation Grant",
                "eligibility": "Residents of districts 4, 7, 8 (flood zones)",
                "deadline": "March 2026"
            },
            {
                "title": "🏭 Industrial Zone Migration Program",
                "description": "Job guarantee and ₫100 million relocation bonus for moving to Hanoi, Hai Phong, or Da Nang manufacturing hubs.",
                "type": "Economic Opportunity",
                "eligibility": "Working-age adults, any experience",
                "deadline": "Rolling admissions"
            },
            {
                "title": "🏠 Climate Adaptation Housing Subsidy",
                "description": "World Bank project providing 18-month rent support and transportation costs for families moving to safer provinces.",
                "type": "Relocation Support",
                "eligibility": "Low-income households in high-risk areas",
                "deadline": "Ongoing program"
            }
        ],
        "Kenya - Nairobi": [
            {
                "title": "🌍 Green Belt Resettlement Initiative",
                "description": "Government program offering KSh 500,000 ($3,800) and 0.25 acre plots in Nakuru or Nyeri for Nairobi slum residents in flood-prone areas.",
                "type": "Relocation Grant",
                "eligibility": "Mathare, Kibera, Mukuru residents",
                "deadline": "September 2026"
            },
            {
                "title": "🌱 Agri-preneurship Migration Scheme",
                "description": "FAO-backed program with farming training and KSh 300,000 startup capital for urban-to-rural migration.",
                "type": "Economic Opportunity",
                "eligibility": "Age 18-60, willing to farm",
                "deadline": "January 2026"
            },
            {
                "title": "🏘️ Affordable Housing Voucher",
                "description": "County government vouchers covering 12-month rent in climate-safe estates (Kiambu, Machakos, Kajiado).",
                "type": "Relocation Support",
                "eligibility": "Flood-affected informal settlement dwellers",
                "deadline": "Active year-round"
            }
        ],
        "Nigeria - Lagos": [
            {
                "title": "🏙️ Abuja Corridor Relocation Grant",
                "description": "Federal program offering ₦5 million ($6,500) and housing priority in new Abuja suburbs for Lagos coastal residents.",
                "type": "Relocation Grant",
                "eligibility": "Victoria Island, Lekki, Apapa residents",
                "deadline": "July 2026"
            },
            {
                "title": "💼 Tech Hub Migration Initiative",
                "description": "IT training and ₦2 million relocation support for moving to emerging tech cities (Kano, Enugu, Port Harcourt).",
                "type": "Economic Opportunity",
                "eligibility": "Age 18-45, basic computer skills",
                "deadline": "Quarterly intake"
            },
            {
                "title": "🏠 NEMA Disaster Resettlement",
                "description": "Emergency management agency providing immediate shelter and 6-month subsidy for families relocating from flood zones.",
                "type": "Relocation Support",
                "eligibility": "Registered flood victims",
                "deadline": "Emergency program"
            }
        ],
        "Brazil - São Paulo": [
            {
                "title": "🌳 Interior Migration Incentive",
                "description": "State program offering R$30,000 ($6,000) and priority housing in inland cities like Campinas or Ribeirão Preto.",
                "type": "Relocation Grant",
                "eligibility": "Residents of flood-risk favelas",
                "deadline": "November 2025"
            },
            {
                "title": "🌾 Rural Development Grant",
                "description": "Federal initiative with agricultural training and R$20,000 for urban-to-rural migration in Minas Gerais or Goiás.",
                "type": "Economic Opportunity",
                "eligibility": "Families interested in farming",
                "deadline": "February 2026"
            },
            {
                "title": "🏘️ Minha Casa Verde Program",
                "description": "Green housing subsidy covering relocation costs and 12-month rent in climate-resilient developments.",
                "type": "Relocation Support",
                "eligibility": "Income <R$2,000/month",
                "deadline": "Rolling applications"
            }
        ],
        "Mexico - Mexico City": [
            {
                "title": "🏔️ Regional Dispersal Program",
                "description": "Federal program offering MXN$150,000 ($8,800) for relocating to mid-sized cities (Querétaro, Guanajuato, Aguascalientes).",
                "type": "Relocation Grant",
                "eligibility": "Residents of high-risk colonias",
                "deadline": "December 2025"
            },
            {
                "title": "🏭 Northern Industry Migration",
                "description": "Manufacturing sector partnership with job placement and MXN$80,000 relocation bonus for moving to Monterrey/Tijuana.",
                "type": "Economic Opportunity",
                "eligibility": "Working-age adults",
                "deadline": "Ongoing recruitment"
            },
            {
                "title": "🏠 SEDATU Housing Assistance",
                "description": "Urban development ministry providing transportation and 9-month rent subsidy for families moving to safer states.",
                "type": "Relocation Support",
                "eligibility": "Flood/landslide affected residents",
                "deadline": "Active during disasters"
            }
        ]
    }
    
    # Default opportunities for regions not in the map
    default_opportunities = [
        {
            "title": "🌍 Global Climate Migration Network",
            "description": f"International program connecting {region_name} residents with safer regions globally. Includes settlement support and cultural integration assistance.",
            "type": "Relocation Support",
            "eligibility": "All climate-displaced persons",
            "deadline": "Open enrollment"
        },
        {
            "title": "💼 ILO Fair Transition Program",
            "description": "Skills training and job placement assistance for workers affected by climate change, with relocation support up to $5,000.",
            "type": "Economic Opportunity",
            "eligibility": "Age 18-65, documented employment",
            "deadline": "Quarterly cohorts"
        },
        {
            "title": "🏘️ UN-Habitat Safe Shelter Initiative",
            "description": "Emergency housing and medium-term accommodation in climate-resilient areas, with livelihood support for 12 months.",
            "type": "Relocation Grant",
            "eligibility": "Verified climate refugees",
            "deadline": "Emergency response - ongoing"
        }
    ]
    
    return opportunities_map.get(region_name, default_opportunities)
