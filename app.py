import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import psycopg2
from sqlalchemy import create_engine
import numpy as np
from datetime import datetime, date
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="WHO AFRO Influenza Landscape Survey",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for WHO styling
st.markdown("""
<style>
    .main-header {
        background-color: #0093D5;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background-color: #F2F2F2;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #0093D5;
        margin-bottom: 1rem;
    }
    
    .sidebar .sidebar-content {
        background-color: #003C71;
    }
    
    .stSelectbox > div > div {
        background-color: #F2F2F2;
    }
    
    .who-blue {
        color: #0093D5;
    }
    
    .who-dark-blue {
        color: #003C71;
    }
    
    .section-header {
        background-color: #003C71;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Database configuration
@st.cache_resource
def init_connection():
    """Initialize database connection"""
    try:
        # Try to get credentials from secrets first, then fallback to environment
        if hasattr(st, 'secrets') and 'database' in st.secrets:
            DB_CONFIG = {
                'host': st.secrets.database.get("DB_HOST", "localhost"),
                'database': st.secrets.database.get("DB_NAME", "who_afro_db"),
                'user': st.secrets.database.get("DB_USER", "postgres"),
                'password': st.secrets.database.get("DB_PASSWORD", "password"),
                'port': st.secrets.database.get("DB_PORT", "5432")
            }
        else:
            # Fallback configuration for development
            DB_CONFIG = {
                'host': "localhost",
                'database': "who_afro_db",
                'user': "postgres",
                'password': "password",
                'port': "5432"
            }
        
        engine = create_engine(
            f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        )
        # Test connection
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        return engine
    except Exception as e:
        st.warning(f"Database connection failed: {e}. Using sample data.")
        return None

@st.cache_data
def load_health_data():
    """Load health indicators data"""
    engine = init_connection()
    if engine:
        try:
            query = """
            SELECT 
                country_name as country,
                life_expectancy,
                infant_mortality_rate as infant_mortality,
                maternal_mortality_rate as maternal_mortality,
                vaccination_coverage,
                cancer_screening_rate as cancer_screening,
                tobacco_use_prevalence as tobacco_prevalence,
                survey_year as year
            FROM health_indicators 
            WHERE survey_year >= 2018
            ORDER BY country_name, survey_year
            """
            return pd.read_sql(query, engine)
        except Exception as e:
            st.warning(f"Failed to load health data: {e}. Using sample data.")
            return generate_sample_health_data()
    else:
        return generate_sample_health_data()

@st.cache_data
def load_vaccination_data():
    """Load vaccination time series data"""
    engine = init_connection()
    if engine:
        try:
            query = """
            SELECT 
                country_name as country,
                survey_year as year,
                mortality_rate,
                dpt_vaccination_rate as vaccination_dpt,
                measles_vaccination_rate as vaccination_measles,
                polio_vaccination_rate as vaccination_polio,
                bcg_vaccination_rate as vaccination_bcg,
                hepatitis_vaccination_rate as vaccination_hepatitis
            FROM vaccination_data 
            WHERE survey_year BETWEEN 2018 AND 2023
            ORDER BY country_name, survey_year
            """
            return pd.read_sql(query, engine)
        except Exception as e:
            st.warning(f"Failed to load vaccination data: {e}. Using sample data.")
            return generate_sample_vaccination_data()
    else:
        return generate_sample_vaccination_data()

@st.cache_data
def load_influenza_data():
    """Load influenza surveillance data"""
    engine = init_connection()
    if engine:
        try:
            query = """
            SELECT 
                country_name as country,
                survey_year as year,
                surveillance_system_exists,
                laboratories_count,
                sentinel_sites_count,
                seasonal_vaccination_policy,
                pandemic_preparedness_score,
                influenza_cases_reported,
                hospitalization_rate
            FROM influenza_surveillance 
            WHERE survey_year >= 2018
            ORDER BY country_name, survey_year
            """
            return pd.read_sql(query, engine)
        except Exception as e:
            st.warning(f"Failed to load influenza data: {e}. Using sample data.")
            return generate_sample_influenza_data()
    else:
        return generate_sample_influenza_data()

def generate_sample_health_data():
    """Generate sample health data for demonstration"""
    countries = ['Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 
                'Cameroon', 'Cape Verde', 'Chad', 'Congo', 'Côte d\'Ivoire', 'Egypt',
                'Ethiopia', 'Gabon', 'Ghana', 'Kenya', 'Madagascar', 'Mali', 'Morocco',
                'Nigeria', 'Rwanda', 'Senegal', 'South Africa', 'Tanzania', 'Uganda']
    
    np.random.seed(42)
    
    health_data = []
    for country in countries:
        for year in range(2018, 2024):
            health_data.append({
                'country': country,
                'life_expectancy': np.random.normal(65, 8),
                'infant_mortality': np.random.exponential(25),
                'maternal_mortality': np.random.exponential(200),
                'vaccination_coverage': np.random.normal(75, 15),
                'cancer_screening': np.random.normal(45, 20),
                'tobacco_prevalence': np.random.normal(20, 8),
                'year': year
            })
    
    return pd.DataFrame(health_data)

def generate_sample_vaccination_data():
    """Generate sample vaccination data for demonstration"""
    countries = ['Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina Faso']  # Subset for performance
    years = list(range(2018, 2024))
    
    np.random.seed(42)
    vaccination_data = []
    
    for country in countries:
        for year in years:
            vaccination_data.append({
                'country': country,
                'year': year,
                'mortality_rate': np.random.normal(8, 1.5),
                'vaccination_dpt': np.random.normal(85, 10),
                'vaccination_measles': np.random.normal(80, 12),
                'vaccination_polio': np.random.normal(88, 8),
                'vaccination_bcg': np.random.normal(75, 15),
                'vaccination_hepatitis': np.random.normal(82, 10)
            })
    
    return pd.DataFrame(vaccination_data)

def generate_sample_influenza_data():
    """Generate sample influenza surveillance data"""
    countries = ['Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina Faso']
    years = list(range(2018, 2024))
    
    np.random.seed(42)
    influenza_data = []
    
    for country in countries:
        for year in years:
            influenza_data.append({
                'country': country,
                'year': year,
                'surveillance_system_exists': np.random.choice([True, False], p=[0.7, 0.3]),
                'laboratories_count': np.random.poisson(3),
                'sentinel_sites_count': np.random.poisson(8),
                'seasonal_vaccination_policy': np.random.choice([True, False], p=[0.6, 0.4]),
                'pandemic_preparedness_score': np.random.normal(60, 20),
                'influenza_cases_reported': np.random.poisson(1500),
                'hospitalization_rate': np.random.exponential(15)
            })
    
    return pd.DataFrame(influenza_data)

# Main application
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌍 WHO AFRO Respiratory Surveillance Dashboard</h1>
        <p>Influenza Landscape Survey for African Region</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar filters
    st.sidebar.markdown("### 🔧 Dashboard Filters")
    
    # Load data
    health_data = load_health_data()
    vaccination_data = load_vaccination_data()
    influenza_data = load_influenza_data()
    
    # Country filter
    countries = ['All Countries'] + sorted(health_data['country'].unique().tolist())
    selected_country = st.sidebar.selectbox("Select Country", countries)
    
    # Date filters
    start_date = st.sidebar.date_input("Start Date", date(2018, 1, 1))
    end_date = st.sidebar.date_input("End Date", date(2023, 12, 31))
    
    # Apply filters button
    if st.sidebar.button("🔄 Apply Filters", type="primary"):
        st.experimental_rerun()
    
    # Filter data based on selection
    if selected_country != 'All Countries':
        filtered_health_data = health_data[health_data['country'] == selected_country]
        filtered_vaccination_data = vaccination_data[vaccination_data['country'] == selected_country]
        filtered_influenza_data = influenza_data[influenza_data['country'] == selected_country]
    else:
        filtered_health_data = health_data
        filtered_vaccination_data = vaccination_data
        filtered_influenza_data = influenza_data
    
    # Key metrics summary
    st.markdown('<div class="section-header"><h2>📊 Key Survey Indicators Summary</h2></div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_life_exp = filtered_health_data['life_expectancy'].mean()
        st.metric("Average Life Expectancy", f"{avg_life_exp:.1f} years", delta=f"{avg_life_exp-65:.1f}")
    
    with col2:
        avg_infant_mort = filtered_health_data['infant_mortality'].mean()
        st.metric("Infant Mortality Rate", f"{avg_infant_mort:.1f} per 1,000", delta=f"{25-avg_infant_mort:.1f}")
    
    with col3:
        avg_vaccination = filtered_health_data['vaccination_coverage'].mean()
        st.metric("Vaccination Coverage", f"{avg_vaccination:.1f}%", delta=f"{avg_vaccination-75:.1f}")
    
    with col4:
        avg_screening = filtered_health_data['cancer_screening'].mean()
        st.metric("Cancer Screening Rate", f"{avg_screening:.1f}%", delta=f"{avg_screening-45:.1f}")
    
    # Main dashboard sections
    st.markdown("---")
    
    # Mortality & Life Expectancy Section
    with st.expander("💀 Mortality & Life Expectancy", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Life Expectancy by Country")
            fig_map = px.choropleth(
                filtered_health_data,
                locations='country',
                color='life_expectancy',
                locationmode='country names',
                scope='africa',
                color_continuous_scale='Blues',
                title="Life Expectancy Distribution"
            )
            fig_map.update_layout(height=400, font=dict(family="Montserrat"))
            st.plotly_chart(fig_map, use_container_width=True)
        
        with col2:
            st.subheader("Mortality Trends Over Time")
            if not filtered_vaccination_data.empty:
                fig_line = px.line(
                    filtered_vaccination_data.groupby('year')['mortality_rate'].mean().reset_index(),
                    x='year',
                    y='mortality_rate',
                    title="Average Mortality Rate Trends",
                    color_discrete_sequence=['#0093D5']
                )
                fig_line.update_layout(height=400, font=dict(family="Montserrat"))
                st.plotly_chart(fig_line, use_container_width=True)
    
    # Vaccination Coverage Section
    with st.expander("💉 Vaccination Coverage", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Vaccination Coverage by Country")
            fig_vacc_map = px.choropleth(
                filtered_health_data,
                locations='country',
                color='vaccination_coverage',
                locationmode='country names',
                scope='africa',
                color_continuous_scale='Greens',
                title="Vaccination Coverage Distribution"
            )
            fig_vacc_map.update_layout(height=400, font=dict(family="Montserrat"))
            st.plotly_chart(fig_vacc_map, use_container_width=True)
        
        with col2:
            st.subheader("Vaccination Types Comparison")
            if not filtered_vaccination_data.empty:
                vacc_data = filtered_vaccination_data.groupby('year')[
                    ['vaccination_dpt', 'vaccination_measles', 'vaccination_polio', 
                     'vaccination_bcg', 'vaccination_hepatitis']
                ].mean().reset_index()
                
                fig_vacc_bar = go.Figure()
                vaccines = ['DPT', 'Measles', 'Polio', 'BCG', 'Hepatitis B']
                colors = ['#0093D5', '#003C71', '#6E6E6E', '#F2F2F2', '#87CEEB']
                
                for i, vacc in enumerate(['vaccination_dpt', 'vaccination_measles', 'vaccination_polio', 
                                        'vaccination_bcg', 'vaccination_hepatitis']):
                    fig_vacc_bar.add_trace(go.Bar(
                        name=vaccines[i],
                        x=vacc_data['year'],
                        y=vacc_data[vacc],
                        marker_color=colors[i]
                    ))
                
                fig_vacc_bar.update_layout(
                    title="Vaccination Coverage by Type",
                    barmode='group',
                    height=400,
                    font=dict(family="Montserrat")
                )
                st.plotly_chart(fig_vacc_bar, use_container_width=True)
    
    # Nutrition & Food Security Section
    with st.expander("🍎 Nutrition & Food Security", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Tobacco Prevalence")
            fig_tobacco = px.scatter(
                filtered_health_data,
                x='life_expectancy',
                y='tobacco_prevalence',
                size='vaccination_coverage',
                color='infant_mortality',
                hover_name='country',
                title="Tobacco Use vs Life Expectancy",
                color_continuous_scale='Reds'
            )
            fig_tobacco.update_layout(height=400, font=dict(family="Montserrat"))
            st.plotly_chart(fig_tobacco, use_container_width=True)
        
        with col2:
            st.subheader("Health Indicators Distribution")
            # Create nutrition status pie chart (simulated data)
            nutrition_data = ['Normal', 'Underweight', 'Overweight', 'Obese']
            values = [68, 15, 12, 5]
            colors = ['#0093D5', '#003C71', '#6E6E6E', '#F2F2F2']
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=nutrition_data,
                values=values,
                marker=dict(colors=colors)
            )])
            fig_pie.update_layout(
                title="Nutrition Status Distribution",
                height=400,
                font=dict(family="Montserrat")
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
    # Data tables section
    st.markdown("---")
    st.markdown('<div class="section-header"><h2>📋 Detailed Data Tables</h2></div>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Health Outcomes", "Time Series Data"])
    
    with tab1:
        st.dataframe(
            filtered_health_data.round(2),
            use_container_width=True,
            height=300
        )
        
        # Download button
        csv = filtered_health_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Health Data as CSV",
            data=csv,
            file_name=f"who_afro_health_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with tab2:
        if not filtered_vaccination_data.empty:
            st.dataframe(
                filtered_vaccination_data.round(2),
                use_container_width=True,
                height=300
            )
            
            # Download button
            csv_ts = filtered_vaccination_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Vaccination Data as CSV",
                data=csv_ts,
                file_name=f"who_afro_vaccination_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6E6E6E; font-size: 0.8rem;">
        <p>WHO AFRO Influenza Landscape Survey Dashboard | Data updated: {}</p>
        <p>World Health Organization - African Region</p>
    </div>
    """.format(datetime.now().strftime("%B %d, %Y")), unsafe_allow_html=True)

if __name__ == "__main__":
    main()