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
                'database': st.secrets.database.get("DB_NAME", "countryprofiles"),
                'user': st.secrets.database.get("DB_USER", "postgres"),
                'password': st.secrets.database.get("DB_PASSWORD", "password"),
                'port': st.secrets.database.get("DB_PORT", "5432")
            }
        else:
            # Fallback configuration for development
            DB_CONFIG = {
                'host': "localhost",
                'database': "countryprofiles",  # Updated to match the query schema
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
        st.warning(f"Database connection failed: {e}. Using CSV data fallback.")
        return None

@st.cache_data
def load_countries_data():
    """Query 1: Load countries data"""
    engine = init_connection()
    if engine:
        try:
            query = "SELECT * FROM countryprofiles.countries;"
            return pd.read_sql(query, engine)
        except Exception as e:
            st.warning(f"Failed to load countries data: {e}. Using CSV fallback.")
            return load_csv_fallback_countries()
    else:
        return load_csv_fallback_countries()

@st.cache_data  
def load_indicators_data():
    """Query 2: Load indicators data"""
    engine = init_connection()
    if engine:
        try:
            query = "SELECT * FROM countryprofiles.indicators;"
            return pd.read_sql(query, engine)
        except Exception as e:
            st.warning(f"Failed to load indicators data: {e}. Using CSV fallback.")
            return load_csv_fallback_indicators()
    else:
        return load_csv_fallback_indicators()

@st.cache_data
def load_indicator_categories_data():
    """Query 3: Load indicator categories data"""
    engine = init_connection()
    if engine:
        try:
            query = "SELECT * FROM countryprofiles.indicator_categories;"
            return pd.read_sql(query, engine)
        except Exception as e:
            st.warning(f"Failed to load indicator categories data: {e}. Using CSV fallback.")
            return load_csv_fallback_categories()
    else:
        return load_csv_fallback_categories()

@st.cache_data
def load_landscape_survey_data():
    """Query 4: Load main landscape survey data - the key query from attachment"""
    engine = init_connection()
    if engine:
        try:
            query = """
            SELECT ci.countryname as Country,
                   ic.cat_name as Category,
                   ci.indicatorname as Indicator,
                   survey_response as Response 
            FROM countryprofiles.country_indicators ci
            INNER JOIN countryprofiles.indicators ind on ind.indicator_id=ci.indicator_id
            INNER JOIN countryprofiles.indicator_categories ic on ic.category_id=ind.category_id
            ORDER BY ci.countryname,ic.category_id,ci.indicatorname;
            """
            return pd.read_sql(query, engine)
        except Exception as e:
            st.warning(f"Failed to load landscape survey data: {e}. Using CSV data.")
            return load_csv_data()
    else:
        return load_csv_data()

def load_csv_data():
    """Load landscape survey data from CSV file"""
    try:
        data_path = "/Users/wildhorizons/Documents/who_projects/afro_pip_landscapesurvey/data/landscapesurvey102025.csv"
        df = pd.read_csv(data_path)
        # Standardize column names to match the query
        df.columns = ['Country', 'Category', 'Indicator', 'Response']
        return df
    except Exception as e:
        st.error(f"Failed to load CSV data: {e}")
        return pd.DataFrame()

def load_csv_fallback_countries():
    """Generate countries data from CSV"""
    landscape_data = load_csv_data()
    if not landscape_data.empty:
        countries = landscape_data['Country'].unique()
        return pd.DataFrame({
            'country_id': range(1, len(countries) + 1),
            'country_name': countries,
            'country_code': [country[:3].upper() for country in countries]
        })
    return pd.DataFrame()

def load_csv_fallback_indicators():
    """Generate indicators data from CSV"""
    landscape_data = load_csv_data()
    if not landscape_data.empty:
        indicators = landscape_data['Indicator'].unique()
        return pd.DataFrame({
            'indicator_id': range(1, len(indicators) + 1),
            'indicator_name': indicators
        })
    return pd.DataFrame()

def load_csv_fallback_categories():
    """Generate categories data from CSV"""
    landscape_data = load_csv_data()
    if not landscape_data.empty:
        categories = landscape_data['Category'].unique()
        return pd.DataFrame({
            'category_id': range(1, len(categories) + 1),
            'cat_name': categories
        })
    return pd.DataFrame()

def generate_descriptive_statistics(data, category=None):
    """Generate comprehensive descriptive statistics for landscape survey data"""
    if data.empty:
        return {}
    
    if category:
        data = data[data['Category'] == category]
    
    stats = {}
    
    # Convert Response to numeric where possible
    numeric_responses = pd.to_numeric(data['Response'], errors='coerce')
    numeric_data = data[numeric_responses.notna()]
    
    if not numeric_data.empty:
        numeric_values = pd.to_numeric(numeric_data['Response'])
        stats['numeric'] = {
            'count': len(numeric_values),
            'mean': numeric_values.mean(),
            'median': numeric_values.median(),
            'std': numeric_values.std(),
            'min': numeric_values.min(),
            'max': numeric_values.max(),
            'q25': numeric_values.quantile(0.25),
            'q75': numeric_values.quantile(0.75)
        }
    
    # Categorical responses
    categorical_data = data[pd.to_numeric(data['Response'], errors='coerce').isna()]
    if not categorical_data.empty:
        value_counts = categorical_data['Response'].value_counts()
        stats['categorical'] = {
            'unique_values': len(value_counts),
            'most_common': value_counts.head(5).to_dict(),
            'total_responses': len(categorical_data)
        }
    
    # Country coverage
    stats['coverage'] = {
        'countries_with_data': data['Country'].nunique(),
        'total_indicators': data['Indicator'].nunique(),
        'response_rate': len(data) / (data['Country'].nunique() * data['Indicator'].nunique()) if data['Indicator'].nunique() > 0 else 0
    }
    
    return stats

@st.cache_data
def load_health_data():
    """Load health indicators data - adapted for landscape survey"""
    landscape_data = load_landscape_survey_data()
    if landscape_data.empty:
        return pd.DataFrame()
    
    # Filter for health-related categories and convert to compatible format
    health_categories = ['Population and Economy', 'Mortality per 100 000 population', 'Mortality per 1000 live births']
    health_data = landscape_data[landscape_data['Category'].isin(health_categories)]
    
    # Pivot to create a more analysis-friendly format
    pivoted = health_data.pivot_table(
        index=['Country'], 
        columns=['Indicator'], 
        values='Response', 
        aggfunc='first'
    ).reset_index()
    
    # Add a year column (since we don't have temporal data, use current year)
    pivoted['year'] = 2024
    
    return pivoted

@st.cache_data
def load_vaccination_data():
    """Load vaccination data from landscape survey"""
    landscape_data = load_landscape_survey_data()
    if landscape_data.empty:
        return pd.DataFrame()
    
    vaccination_data = landscape_data[landscape_data['Category'] == 'Vaccination']
    
    # Convert to analysis format
    pivoted = vaccination_data.pivot_table(
        index=['Country'], 
        columns=['Indicator'], 
        values='Response', 
        aggfunc='first'
    ).reset_index()
    
    pivoted['year'] = 2024
    return pivoted

@st.cache_data  
def load_surveillance_data():
    """Load surveillance data from landscape survey"""
    landscape_data = load_landscape_survey_data()
    if landscape_data.empty:
        return pd.DataFrame()
    
    surveillance_categories = [
        'Influenza like Illness (ILI) Surveillance',
        'Severe acute respiratory infection (SARI) surveillance',
        'Virological surveillance'
    ]
    
    surveillance_data = landscape_data[landscape_data['Category'].isin(surveillance_categories)]
    
    # Convert to analysis format
    pivoted = surveillance_data.pivot_table(
        index=['Country'], 
        columns=['Indicator'], 
        values='Response', 
        aggfunc='first'
    ).reset_index()
    
    pivoted['year'] = 2024
    return pivoted

@st.cache_data
def load_category_indicators():
    """Load country indicators data stratified by category - matches the provided SQL query"""
    engine = init_connection()
    if engine:
        try:
            query = """
            SELECT 
                ci.countryname as country,
                ic.cat_name as category,
                ci.indicatorname as indicator,
                ci.survey_response as response,
                ci.survey_year as year,
                ic.category_id,
                ind.indicator_id
            FROM countryprofiles.country_indicators ci
            INNER JOIN countryprofiles.indicators ind ON ind.indicator_id = ci.indicator_id
            INNER JOIN countryprofiles.indicator_categories ic ON ic.category_id = ind.category_id
            ORDER BY ci.countryname, ic.category_id, ci.indicatorname
            """
            return pd.read_sql(query, engine)
        except Exception as e:
            st.warning(f"Failed to load category indicators: {e}. Using sample data.")
            return generate_sample_category_data()
    else:
        return generate_sample_category_data()

def generate_sample_category_data():
    """Generate sample category indicators data for demonstration"""
    countries = ['Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 
                'Cameroon', 'Cape Verde', 'Chad', 'Congo', 'Ethiopia', 'Ghana', 'Kenya', 
                'Nigeria', 'Rwanda', 'Senegal', 'South Africa', 'Tanzania', 'Uganda']
    
    categories = [
        'Surveillance & Detection Systems',
        'Laboratory Capacity & Diagnostics', 
        'Pandemic Preparedness & Response',
        'Vaccination & Immunization Programs',
        'Healthcare Infrastructure & Workforce',
        'Health Security & Emergency Response'
    ]
    
    indicators_by_category = {
        'Surveillance & Detection Systems': [
            'National surveillance system exists',
            'Surveillance coverage percentage',
            'Reporting timeliness score',
            'Electronic reporting capability',
            'Outbreak detection capacity'
        ],
        'Laboratory Capacity & Diagnostics': [
            'National laboratories count',
            'Diagnostic capabilities score',
            'Quality assurance programs',
            'Equipment maintenance score',
            'Staff competency level'
        ],
        'Pandemic Preparedness & Response': [
            'Preparedness plan exists',
            'Simulation exercises conducted',
            'Stockpile adequacy score',
            'Response coordination score',
            'International cooperation level'
        ],
        'Vaccination & Immunization Programs': [
            'Routine immunization coverage',
            'Cold chain capacity score',
            'Vaccine stock adequacy',
            'Healthcare worker vaccination rate',
            'Vaccine hesitancy index'
        ],
        'Healthcare Infrastructure & Workforce': [
            'Hospitals per 100k population',
            'ICU beds per 100k population',
            'Doctors per 100k population',
            'Nurses per 100k population',
            'Telemedicine capability score'
        ],
        'Health Security & Emergency Response': [
            'Emergency response time (minutes)',
            'Emergency operations center capacity',
            'Contact tracing system capacity',
            'Risk communication effectiveness',
            'Community engagement score'
        ]
    }
    
    np.random.seed(42)
    category_data = []
    
    for country in countries:
        for year in range(2018, 2024):
            for cat_id, category in enumerate(categories, 1):
                for ind_id, indicator in enumerate(indicators_by_category[category], 1):
                    # Generate appropriate response values based on indicator type
                    if 'percentage' in indicator.lower() or 'coverage' in indicator.lower():
                        response = np.random.normal(75, 15)
                        response = max(0, min(100, response))  # Clamp to 0-100
                    elif 'score' in indicator.lower() or 'level' in indicator.lower():
                        response = np.random.normal(60, 20)
                        response = max(0, min(100, response))  # Clamp to 0-100
                    elif 'count' in indicator.lower() or 'per 100k' in indicator.lower():
                        response = np.random.exponential(10)
                    elif 'exists' in indicator.lower():
                        response = np.random.choice([0, 1], p=[0.3, 0.7])
                    elif 'time' in indicator.lower():
                        response = np.random.exponential(30)
                    else:
                        response = np.random.normal(50, 25)
                    
                    category_data.append({
                        'country': country,
                        'category': category,
                        'indicator': indicator,
                        'response': round(response, 2),
                        'year': year,
                        'category_id': cat_id,
                        'indicator_id': ind_id
                    })
    
    return pd.DataFrame(category_data)

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
        <h1>🌍 WHO AFRO Influenza Landscape Survey Dashboard</h1>
        <p>Country Profiles and Respiratory Surveillance Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar filters
    st.sidebar.markdown("### 🔧 Dashboard Filters")
    
    # Load core data
    landscape_data = load_landscape_survey_data()
    countries_data = load_countries_data()
    categories_data = load_indicator_categories_data()
    
    if landscape_data.empty:
        st.error("No landscape survey data available. Please check your data source.")
        return
    
    # Category filter
    available_categories = sorted(landscape_data['Category'].unique())
    selected_categories = st.sidebar.multiselect(
        "Select Categories", 
        available_categories, 
        default=available_categories[:3]  # Select first 3 by default
    )
    
    # Country filter
    available_countries = sorted(landscape_data['Country'].unique())
    selected_countries = st.sidebar.multiselect(
        "Select Countries", 
        available_countries,
        default=available_countries[:10] if len(available_countries) > 10 else available_countries
    )
    
    # Apply filters
    filtered_data = landscape_data[
        (landscape_data['Category'].isin(selected_categories)) &
        (landscape_data['Country'].isin(selected_countries))
    ]
    
    # Data Quality Overview
    st.markdown('<div class="section-header"><h2>📊 Data Overview</h2></div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_countries = landscape_data['Country'].nunique()
        st.metric("Total Countries", total_countries)
    
    with col2:
        total_categories = landscape_data['Category'].nunique()
        st.metric("Survey Categories", total_categories)
    
    with col3:
        total_indicators = landscape_data['Indicator'].nunique()
        st.metric("Total Indicators", total_indicators)
    
    with col4:
        total_responses = len(landscape_data)
        st.metric("Total Responses", f"{total_responses:,}")
    
    # Category-based Analysis
    st.markdown("---")
    st.markdown('<div class="section-header"><h2>📈 Category-Based Analysis</h2></div>', unsafe_allow_html=True)
    
    # Category selector for detailed analysis
    selected_category = st.selectbox(
        "Select Category for Detailed Analysis",
        available_categories
    )
    
    if selected_category:
        category_data = landscape_data[landscape_data['Category'] == selected_category]
        
        # Category statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"📊 {selected_category} - Descriptive Statistics")
            stats = generate_descriptive_statistics(category_data)
            
            if 'numeric' in stats:
                numeric_stats = stats['numeric']
                st.write("**Numeric Indicators:**")
                stats_df = pd.DataFrame({
                    'Statistic': ['Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Q25', 'Q75'],
                    'Value': [
                        numeric_stats['count'],
                        f"{numeric_stats['mean']:.2f}" if not pd.isna(numeric_stats['mean']) else 'N/A',
                        f"{numeric_stats['median']:.2f}" if not pd.isna(numeric_stats['median']) else 'N/A',
                        f"{numeric_stats['std']:.2f}" if not pd.isna(numeric_stats['std']) else 'N/A',
                        f"{numeric_stats['min']:.2f}" if not pd.isna(numeric_stats['min']) else 'N/A',
                        f"{numeric_stats['max']:.2f}" if not pd.isna(numeric_stats['max']) else 'N/A',
                        f"{numeric_stats['q25']:.2f}" if not pd.isna(numeric_stats['q25']) else 'N/A',
                        f"{numeric_stats['q75']:.2f}" if not pd.isna(numeric_stats['q75']) else 'N/A'
                    ]
                })
                st.dataframe(stats_df, use_container_width=True)
            
            if 'categorical' in stats:
                categorical_stats = stats['categorical']
                st.write("**Categorical Responses:**")
                st.write(f"- Unique values: {categorical_stats['unique_values']}")
                st.write(f"- Total responses: {categorical_stats['total_responses']}")
                
                st.write("**Most common responses:**")
                for value, count in categorical_stats['most_common'].items():
                    st.write(f"- {value}: {count}")
        
        with col2:
            st.subheader(f"🗺️ {selected_category} - Country Coverage")
            
            # Country coverage visualization
            country_coverage = category_data.groupby('Country').size().reset_index(name='Indicator_Count')
            
            if not country_coverage.empty:
                fig_coverage = px.bar(
                    country_coverage.head(20),  # Show top 20 countries
                    x='Country',
                    y='Indicator_Count',
                    title=f"Number of Indicators per Country - {selected_category}",
                    color='Indicator_Count',
                    color_continuous_scale='Blues'
                )
                fig_coverage.update_layout(
                    height=400,
                    xaxis_tickangle=-45,
                    font=dict(family="Montserrat")
                )
                st.plotly_chart(fig_coverage, use_container_width=True)
    
    # Cross-Category Analysis  
    st.markdown("---")
    st.markdown('<div class="section-header"><h2>🔗 Cross-Category Analysis</h2></div>', unsafe_allow_html=True)
    
    # Category comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Indicators per Category")
        category_counts = landscape_data.groupby('Category')['Indicator'].nunique().reset_index()
        category_counts = category_counts.sort_values('Indicator', ascending=False)
        
        fig_cat = px.bar(
            category_counts,
            x='Indicator',
            y='Category',
            orientation='h',
            title="Number of Indicators by Category",
            color='Indicator',
            color_continuous_scale='Viridis'
        )
        fig_cat.update_layout(height=400, font=dict(family="Montserrat"))
        st.plotly_chart(fig_cat, use_container_width=True)
    
    with col2:
        st.subheader("🌍 Country Response Completeness")
        country_completeness = landscape_data.groupby('Country').size().reset_index(name='Total_Responses')
        country_completeness = country_completeness.sort_values('Total_Responses', ascending=False).head(15)
        
        fig_complete = px.bar(
            country_completeness,
            x='Country',
            y='Total_Responses',
            title="Total Survey Responses by Country (Top 15)",
            color='Total_Responses',
            color_continuous_scale='Oranges'
        )
        fig_complete.update_layout(
            height=400,
            xaxis_tickangle=-45,
            font=dict(family="Montserrat")
        )
        st.plotly_chart(fig_complete, use_container_width=True)
    
    # Detailed Data Tables
    st.markdown("---")
    st.markdown('<div class="section-header"><h2>📋 Detailed Survey Data</h2></div>', unsafe_allow_html=True)
    
    # Data explorer
    st.subheader("� Data Explorer")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        filter_country = st.selectbox(
            "Filter by Country",
            ['All'] + sorted(landscape_data['Country'].unique())
        )
    
    with col2:
        filter_category = st.selectbox(
            "Filter by Category", 
            ['All'] + sorted(landscape_data['Category'].unique())
        )
    
    with col3:
        search_indicator = st.text_input("Search Indicators", "")
    
    # Apply filters to data
    display_data = landscape_data.copy()
    
    if filter_country != 'All':
        display_data = display_data[display_data['Country'] == filter_country]
    
    if filter_category != 'All':
        display_data = display_data[display_data['Category'] == filter_category]
    
    if search_indicator:
        display_data = display_data[
            display_data['Indicator'].str.contains(search_indicator, case=False, na=False)
        ]
    
    # Display filtered data
    st.dataframe(
        display_data,
        use_container_width=True,
        height=400
    )
    
    # Download options
    col1, col2 = st.columns(2)
    
    with col1:
        csv_data = display_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv_data,
            file_name=f"landscape_survey_filtered_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col2:
        full_csv = landscape_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Complete Dataset as CSV",
            data=full_csv,
            file_name=f"landscape_survey_complete_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    # Limitations and Notes
    st.markdown("---")
    st.markdown('<div class="section-header"><h2>⚠️ Data Limitations & Notes</h2></div>', unsafe_allow_html=True)
    
    st.warning("""
    **Current Data Limitations:**
    
    1. **Temporal Analysis**: The current dataset appears to be cross-sectional (single time point), limiting trend analysis capabilities.
    
    2. **Data Types**: Mixed data types (numeric and categorical) in the Response field require careful handling for statistical analysis.
    
    3. **Missing Values**: Some countries may have incomplete responses across categories, affecting comparative analysis.
    
    4. **Standardization**: Response formats vary across indicators, making direct numerical comparisons challenging.
    
    5. **Geographic Coverage**: Analysis is limited to countries present in the dataset - may not represent all WHO AFRO member states.
    
    6. **Indicator Definitions**: Without metadata about indicator definitions and measurement units, interpretation may be limited.
    """)
    
if __name__ == "__main__":
    main()