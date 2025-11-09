"""
ClimaShield 2.0 - Main Application Entry Point
AI-powered climate displacement risk prediction and micro-insurance platform.
"""

import streamlit as st
import ui
import data_model


# Page configuration
st.set_page_config(
    page_title="ClimaShield",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .tagline {
        text-align: center;
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application function."""
    
    # Initialize session state
    ui.initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🌍 ClimaShield</h1>', unsafe_allow_html=True)
    st.markdown('<p class="tagline">✨ AI for Predictive Protection & Dignified Adaptation</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/5208/5208410.png", width=100)
        st.title("ClimaShield")
        st.markdown("---")
        
        st.markdown("""
        ### About
        ClimaShield uses **AI and open climate data** to:
        
        - 🗺️ Predict displacement risk
        - 💰 Calculate adaptive insurance premiums
        - 🚨 Send early warnings
        - 🤝 Promote climate resilience
        
        ### Data Source
        **AI for Water and Planet (AIWP)**  
        Open satellite and sensor data for climate analysis.
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("📅 November 2025")
    
    # Main content
    try:
        # Initialize data model with loading message
        with st.spinner("🔄 Loading AIWP climate data and training AI model..."):
            df_predictions, model, scaler, feature_cols = data_model.initialize_data_model()
        
        st.success("✅ AI model trained successfully on AIWP open data!")
        
        # Display statistics dashboard
        ui.render_stats_dashboard(df_predictions)
        
        st.markdown("---")
        
        # Map section with region selection
        ui.render_map_section(df_predictions)
        
        st.markdown("---")
        
        # Alert system
        ui.render_alert_system(df_predictions)
        
        st.markdown("---")
        
        # Premium calculator
        ui.render_premium_calculator(df_predictions)
        
        st.markdown("---")
        
        # AI Explainability
        ui.render_ai_explainability(df_predictions, model, feature_cols)
        
        st.markdown("---")
        
        # Footer
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0; color: #666;">
            <p><strong>ClimaShield 2.0</strong> - Empowering Communities with AI-Driven Climate Resilience</p>
            <p>Built with ❤️ using Streamlit, scikit-learn, and AIWP Open Data</p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ Error initializing application: {e}")
        st.info("Please check your data connection and dependencies.")

if __name__ == "__main__":
    main()
