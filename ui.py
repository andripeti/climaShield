"""
ClimaShield 2.0 - User Interface Components
Streamlit UI layout, map visualization, and interactive components.
"""

import streamlit as st
import pandas as pd
import folium
from folium import plugins
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from typing import Optional
import utils
import data_model


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "selected_region" not in st.session_state:
        st.session_state.selected_region = None
    
    if "alert_log" not in st.session_state:
        st.session_state.alert_log = []
    
    if "map_click" not in st.session_state:
        st.session_state.map_click = None


def create_risk_heatmap(df: pd.DataFrame, selected_region: Optional[str] = None) -> folium.Map:
    """
    Create interactive Folium map with risk heatmap visualization.
    
    Args:
        df: DataFrame with predictions and coordinates
        selected_region: Currently selected region name
    
    Returns:
        Folium Map object
    """
    # Center map on mean coordinates
    center_lat = df["lat"].mean()
    center_lon = df["lon"].mean()
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=2,
        tiles="OpenStreetMap"
    )
    
    # Add markers for each region
    for _, row in df.iterrows():
        region_name = row["region"]
        risk_score = row["predicted_risk_score"]
        lat, lon = row["lat"], row["lon"]
        
        # Get risk category and color
        category, color = utils.get_risk_category(risk_score)
        
        # Generate summary
        summary = utils.generate_risk_summary(risk_score, region_name)
        
        # Highlight selected region
        is_selected = (region_name == selected_region)
        
        # Create marker with custom styling
        icon_color = color if not is_selected else "#8e44ad"  # Purple for selected
        radius = 15 if is_selected else 10
        
        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            popup=folium.Popup(
                f"""
                <div style="font-family: Arial; width: 250px;">
                    <h4 style="color: {color};">{region_name}</h4>
                    <p><b>Risk Score:</b> {risk_score:.3f}</p>
                    <p><b>Category:</b> {category}</p>
                    <hr>
                    <p style="font-size: 12px;">{summary}</p>
                </div>
                """,
                max_width=300
            ),
            tooltip=f"{region_name}: {category}",
            color=icon_color,
            fill=True,
            fillColor=icon_color,
            fillOpacity=0.7 if is_selected else 0.5,
            weight=3 if is_selected else 2
        ).add_to(m)
    
    # Add fullscreen button
    plugins.Fullscreen().add_to(m)
    
    return m


def render_map_section(df: pd.DataFrame):
    """
    Render the interactive map section with region synchronization.
    
    Args:
        df: DataFrame with predictions
    """
    st.subheader("🗺️ AI Risk Heatmap")
    st.write("Interactive map showing climate displacement risk across regions. Click on a marker to select a region.")
    
    # Create two columns for dropdown and map
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Region dropdown
        region_list = ["Select a region..."] + sorted(df["region"].tolist())
        
        # Set default index based on session state
        default_index = 0
        if st.session_state.selected_region and st.session_state.selected_region in region_list:
            default_index = region_list.index(st.session_state.selected_region)
        
        selected_dropdown = st.selectbox(
            "Select Region:",
            region_list,
            index=default_index,
            key="region_dropdown"
        )
        
        # Update session state if dropdown changed
        if selected_dropdown != "Select a region...":
            if st.session_state.selected_region != selected_dropdown:
                st.session_state.selected_region = selected_dropdown
                st.rerun()
        
        # Display region info if selected
        if st.session_state.selected_region:
            region_data = data_model.get_region_data(df, st.session_state.selected_region)
            if region_data is not None:
                st.markdown("---")
                st.markdown(f"**Region:** {region_data['region']}")
                st.markdown(f"**Country:** {region_data['country']}")
                st.markdown(f"**Risk Score:** {region_data['predicted_risk_score']:.3f}")
                
                category, color = utils.get_risk_category(region_data['predicted_risk_score'])
                st.markdown(f"**Category:** <span style='color:{color}; font-weight:bold;'>{category}</span>", unsafe_allow_html=True)
    
    with col2:
        # Create and render map
        risk_map = create_risk_heatmap(df, st.session_state.selected_region)
        
        # Render map with interaction
        map_data = st_folium(
            risk_map,
            width=800,
            height=500,
            returned_objects=["last_object_clicked"]
        )
        
        # Handle map clicks (would require custom JavaScript in production)
        # For now, users can click markers to see popups


def render_premium_calculator(df: pd.DataFrame):
    """
    Render AI-powered insurance premium calculator.
    
    Args:
        df: DataFrame with predictions
    """
    st.subheader("💰 AI-Powered Premium Calculator")
    st.write("Get personalized insurance premium based on your region's AI-predicted risk and your income level.")
    
    # Input columns
    col1, col2 = st.columns(2)
    
    with col1:
        # Income input
        income = st.number_input(
            "Monthly Income (USD):",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100,
            help="Your monthly household income in USD"
        )
    
    with col2:
        # Region selection (synced with map)
        if st.session_state.selected_region:
            st.info(f"📍 Selected Region: **{st.session_state.selected_region}**")
        else:
            st.warning("⚠️ Please select a region from the map above")
    
    # Calculate button
    if st.button("Calculate Premium", type="primary", disabled=(st.session_state.selected_region is None)):
        if st.session_state.selected_region:
            # Get region data
            region_data = data_model.get_region_data(df, st.session_state.selected_region)
            
            if region_data is not None:
                risk_score = region_data["predicted_risk_score"]
                
                # Calculate premium
                premium_data = utils.calculate_ai_premium(
                    risk_score=risk_score,
                    income=income,
                    base_rate=50.0
                )
                
                # Display results
                st.markdown("---")
                st.markdown("### 📊 Your Insurance Quote")
                
                # Create three columns for key metrics
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.metric(
                        "Monthly Premium",
                        utils.format_currency(premium_data["monthly_premium"]),
                        help="AI-adjusted monthly premium based on risk and income"
                    )
                
                with metric_col2:
                    st.metric(
                        "Annual Premium",
                        utils.format_currency(premium_data["annual_premium"]),
                        help="Total annual cost"
                    )
                
                with metric_col3:
                    st.metric(
                        "Estimated Payout",
                        utils.format_currency(premium_data["estimated_payout"]),
                        help="Estimated compensation in case of climate disaster"
                    )
                
                # Detailed breakdown
                with st.expander("📋 View Detailed Breakdown"):
                    st.markdown(f"""
                    - **Base Rate:** {utils.format_currency(50.0)}
                    - **Risk Score:** {premium_data['risk_score']}
                    - **Income Index:** {premium_data['income_index']}
                    - **Premium Multiplier:** {premium_data['premium_multiplier']}
                    - **Subsidy Amount:** {utils.format_currency(premium_data['subsidy_amount'])}
                    """)
                    
                    st.info("💡 **How it works:** The AI model adjusts your premium based on regional climate risk and your income level. Lower income households receive higher subsidies for affordable coverage.")
                
                # Visualize premium breakdown
                fig = go.Figure(data=[
                    go.Bar(
                        x=["Base Rate", "Risk Adjustment", "Income Subsidy", "Final Premium"],
                        y=[50.0, 50.0 * premium_data["premium_multiplier"], -premium_data["subsidy_amount"], premium_data["monthly_premium"]],
                        marker_color=["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]
                    )
                ])
                fig.update_layout(
                    title="Premium Calculation Breakdown",
                    yaxis_title="Amount (USD)",
                    showlegend=False,
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)


def render_alert_system(df: pd.DataFrame):
    """
    Render smart alert system with risk warnings.
    
    Args:
        df: DataFrame with predictions
    """
    st.subheader("🚨 Smart Alert System")
    
    # Check if selected region has high risk
    if st.session_state.selected_region:
        region_data = data_model.get_region_data(df, st.session_state.selected_region)
        
        if region_data is not None:
            risk_score = region_data["predicted_risk_score"]
            
            # Check alert threshold
            if utils.check_alert_threshold(risk_score, threshold=0.5):
                alert_message = utils.generate_alert_message(risk_score, st.session_state.selected_region)
                
                # Display alert
                if risk_score >= 0.7:
                    st.error(alert_message)
                else:
                    st.warning(alert_message)
                
                # Add to alert log
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                alert_entry = {
                    "timestamp": timestamp,
                    "region": st.session_state.selected_region,
                    "risk_score": risk_score,
                    "message": alert_message
                }
                
                # Avoid duplicate alerts
                if not any(a["region"] == st.session_state.selected_region for a in st.session_state.alert_log):
                    st.session_state.alert_log.append(alert_entry)
                
                # Action buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📋 View Insurance Options", key="alert_insurance"):
                        st.info("Scroll down to the Premium Calculator to get your personalized quote!")
                
                with col2:
                    if st.button("📱 Simulate SMS Alert", key="alert_sms"):
                        st.success(f"✅ SMS alert sent to registered number!\n\nMessage: {alert_message}")
            else:
                st.success(f"✅ {st.session_state.selected_region} is currently at acceptable risk levels.")
    else:
        st.info("Select a region from the map to check for alerts.")
    
    # Alert log
    if st.session_state.alert_log:
        with st.expander(f"📜 Alert Log ({len(st.session_state.alert_log)} alerts)"):
            for alert in reversed(st.session_state.alert_log):
                st.markdown(f"""
                **{alert['timestamp']}** - {alert['region']} (Risk: {alert['risk_score']:.3f})  
                {alert['message']}
                """)
                st.markdown("---")


def render_ai_explainability(df: pd.DataFrame, model, feature_cols: list):
    """
    Render AI explainability section with feature importance.
    
    Args:
        df: DataFrame with predictions
        model: Trained ML model
        feature_cols: List of feature names
    """
    st.subheader("🧠 Explain AI Predictions")
    st.write("Understand how the AI model makes risk predictions based on climate features.")
    
    # Get feature importance
    importance_df = data_model.get_feature_importance(model, feature_cols)
    
    # Display top features
    st.markdown("### Top Risk Factors")
    
    fig = px.bar(
        importance_df.head(8),
        x="importance",
        y="feature",
        orientation="h",
        title="Feature Importance in Risk Prediction",
        labels={"importance": "Importance Score", "feature": "Climate Feature"},
        color="importance",
        color_continuous_scale="Reds"
    )
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Explanation text
    with st.expander("ℹ️ How to Interpret"):
        st.markdown("""
        **Feature importance** shows which climate factors contribute most to risk predictions:
        
        - **Higher values** = More influential in determining risk
        - **Precipitation, temperature anomalies, and flood history** typically have the strongest impact
        - **Vegetation index and soil moisture** indicate resilience factors
        
        The Random Forest model analyzes these features together to predict displacement risk.
        """)


def render_stats_dashboard(df: pd.DataFrame):
    """
    Render statistics dashboard with key metrics.
    
    Args:
        df: DataFrame with predictions
    """
    st.subheader("📈 Global Risk Statistics")
    
    # Calculate statistics
    avg_risk = df["predicted_risk_score"].mean()
    high_risk_count = len(df[df["predicted_risk_score"] >= 0.6])
    total_regions = len(df)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Regions Monitored", total_regions)
    
    with col2:
        st.metric("Average Risk Score", f"{avg_risk:.3f}")
    
    with col3:
        st.metric("High Risk Regions", high_risk_count)
    
    with col4:
        pct_high_risk = (high_risk_count / total_regions) * 100
        st.metric("High Risk %", f"{pct_high_risk:.1f}%")
    
    # Risk distribution chart
    st.markdown("### Risk Distribution Across Regions")
    
    fig = px.histogram(
        df,
        x="predicted_risk_score",
        nbins=20,
        title="Distribution of Risk Scores",
        labels={"predicted_risk_score": "Risk Score", "count": "Number of Regions"},
        color_discrete_sequence=["#e74c3c"]
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
