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
from urllib.parse import quote_plus
import os
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="WHO AFRO Influenza Landscape Survey",
    page_icon="assets/who_logo.png",
    layout="wide"
)

# Custom CSS for WHO styling
st.markdown("""
<style>
    .main-header {
        background-color: #0093D5;
        color: white;
        padding: 1rem;
        width: 100vw;
        margin-left: calc(-50vw + 50%);
        margin-top: 0px;
        border-radius: 0;
        text-align: center;
        margin-bottom: 2rem;
        position: relative;
    }
    
    .metric-card {
        background-color: #F2F2F2;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #0093D5;
        margin-bottom: 1rem;
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
        if hasattr(st, 'secrets'):
            # Check if database section exists, otherwise use top-level secrets
            if 'database' in st.secrets:
                DB_CONFIG = {
                    'host': st.secrets.database.get("DB_HOST", "localhost"),
                    'database': st.secrets.database.get("DB_NAME", "emp_pip"),
                    'user': st.secrets.database.get("DB_USER", "postgres"),
                    'password': st.secrets.database.get("DB_PASSWORD", "password"),
                    'port': st.secrets.database.get("DB_PORT", "5432")
                }
            else:
                # Use top-level secrets
                DB_CONFIG = {
                    'host': st.secrets.get("DB_HOST", "localhost"),
                    'database': st.secrets.get("DB_NAME", "emp_pip"),
                    'user': st.secrets.get("DB_USER", "postgres"),
                    'password': st.secrets.get("DB_PASSWORD", "password"),
                    'port': st.secrets.get("DB_PORT", "5432")
                }
        else:
            # Fallback configuration for development
            DB_CONFIG = {
                'host': "localhost",
                'database': "emp_pip",  # Updated to match your actual database
                'user': "postgres",
                'password': "password",
                'port': "5432"
            }
        
        # Debug: Print connection info (remove in production)
        #st.info(f"🔧 Attempting connection to: {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']} as {DB_CONFIG['user']}")
        
        # URL encode the password to handle special characters like @
        encoded_password = quote_plus(DB_CONFIG['password'])
        
        engine = create_engine(
            f"postgresql://{DB_CONFIG['user']}:{encoded_password}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        )
        # Test connection
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        
        st.success("✅ Database connection successful!")
        return engine
    except Exception as e:
        # Clean warning message without exposing error details - show only once per session
        if 'db_warning_shown' not in st.session_state:
            db_name = DB_CONFIG.get('database', 'emp_pip') if 'DB_CONFIG' in locals() else 'emp_pip'
            #st.warning(f"⚠️ Database '{db_name}' is currently unavailable. Static CSV data is currently being used.")
            st.session_state.db_warning_shown = True
       # st.info("ℹ️ Falling back to CSV data from the data/ folder")
        return None

@st.cache_data
def load_countries_data():
    """Query 1: Load countries data"""
    engine = init_connection()
    if engine:
        try:
            # Try different possible schema/table combinations
            queries_to_try = [
                "SELECT * FROM countryprofiles.countries;",
                "SELECT DISTINCT countryname as country_name FROM countryprofiles.country_indicators;",
            ]
            
            for query in queries_to_try:
                try:
                    st.write(f"🔍 Trying query: {query}")
                    return pd.read_sql(query, engine)
                except Exception as query_error:
                    st.write(f"❌ Query failed: {query_error}")
                    continue
                    
            # If all queries fail
            raise Exception("All country queries failed")
            
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

            queries_to_try = [
                """
                SELECT ci.countryname as country,
                       ic.cat_name as category,
                       ci.indicatorname as indicator,
                       survey_response as response,
                       ic.category_id as category_id,
                       ci.indicator_id as indicator_id
                FROM countryprofiles.country_indicators ci
                INNER JOIN countryprofiles.indicators ind on ind.indicator_id=ci.indicator_id
                INNER JOIN countryprofiles.indicator_categories ic on ic.category_id=ind.category_id
                ORDER BY ci.countryname,ic.category_id,ci.indicatorname;
                """
            ]
            
            for query in queries_to_try:
                try:
                    st.write(f"🔍 Trying main query...")
                    result = pd.read_sql(query, engine)
                    st.success(f"✅ Successfully loaded {len(result)} records from database!")
                    return result
                except Exception as query_error:
                    st.write(f"❌ Database query failed: {query_error}")
                    continue
                    
            # If all queries fail
            raise Exception("All landscape survey queries failed")
            
        except Exception as e:
            st.warning(f"Failed to load landscape survey data: {e}. Using CSV data.")
            return load_csv_data()
    else:
        return load_csv_data()

def load_csv_data():
    """Load landscape survey data from CSV file"""
    try:
        # Use relative path that works in any environment
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, "data", "landscapesurvey2024.csv")
        df = pd.read_csv(data_path)
        # Use the actual column names from the file
        # Columns are: country, category, indicator, response, category_id, indicator_id
        return df
    except Exception as e:
        st.error(f"Failed to load CSV data: {e}")
        return pd.DataFrame()

def load_csv_fallback_countries():
    """Fallback function to load countries from CSV"""
    try:
        landscape_data = load_csv_data()
        countries = landscape_data['country'].unique()
        return [{'name': country} for country in countries]
    except Exception as e:
        st.error(f"Failed to load countries from CSV: {e}")
        return []

def load_csv_fallback_indicators():
    """Fallback function to load indicators from CSV"""
    try:
        landscape_data = load_csv_data()
        indicators = landscape_data['indicator'].unique()
        return [{'name': indicator} for indicator in indicators]
    except Exception as e:
        st.error(f"Failed to load indicators from CSV: {e}")
        return []

def load_csv_fallback_categories():
    """Generate categories data from CSV"""
    landscape_data = load_csv_data()
    if not landscape_data.empty:
        categories = landscape_data['category'].unique()
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
        data = data[data['category'] == category]
    
    stats = {}
    
    # Convert response to numeric where possible
    numeric_responses = pd.to_numeric(data['response'], errors='coerce')
    numeric_data = data[numeric_responses.notna()]
    
    if not numeric_data.empty:
        numeric_values = pd.to_numeric(numeric_data['response'])
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
    categorical_data = data[pd.to_numeric(data['response'], errors='coerce').isna()]
    if not categorical_data.empty:
        value_counts = categorical_data['response'].value_counts()
        stats['categorical'] = {
            'unique_values': len(value_counts),
            'most_common': value_counts.head(5).to_dict(),
            'total_responses': len(categorical_data)
        }
    
    # Country coverage
    stats['coverage'] = {
        'countries_with_data': data['country'].nunique(),
        'total_indicators': data['indicator'].nunique(),
        'response_rate': len(data) / (data['country'].nunique() * data['indicator'].nunique()) if data['indicator'].nunique() > 0 else 0
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
    health_data = landscape_data[landscape_data['category'].isin(health_categories)]
    
    # Pivot to create a more analysis-friendly format
    pivoted = health_data.pivot_table(
        index=['country'], 
        columns=['indicator'], 
        values='response', 
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
    
    vaccination_data = landscape_data[landscape_data['category'] == 'Vaccination']
    
    # Convert to analysis format
    pivoted = vaccination_data.pivot_table(
        index=['country'], 
        columns=['indicator'], 
        values='response', 
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
    
    surveillance_data = landscape_data[landscape_data['category'].isin(surveillance_categories)]
    
    # Convert to analysis format
    pivoted = surveillance_data.pivot_table(
        index=['country'], 
        columns=['indicator'], 
        values='response', 
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
    # WHO Logo and Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        try:
            st.image("assets/who_logo.png", width=150)
        except FileNotFoundError:
            st.write("🌍")  # Fallback emoji if logo not found
    
    st.markdown("""
    <div class="main-header">
        <h1>WHO AFRO Influenza Landscape Survey Dashboard</h1>
        <p>Survey Period: 2023 - 2024</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load core data
    landscape_data = load_landscape_survey_data()
    
    if landscape_data.empty:
        st.error("No landscape survey data available. Please check your data source.")
        return

    # Regional Demographics and Economic Overview
    st.markdown('<div class="section-header"><h2>🌍 Regional Demographics and Economic Overview</h2></div>', unsafe_allow_html=True)
    
    # Category filter
    available_categories = sorted(landscape_data['category'].unique())
    selected_categories = st.sidebar.multiselect(
        "Select Categories", 
        available_categories, 
        default=[]  # No categories selected by default
    )
    
    # Country filter
    available_countries = sorted(landscape_data['country'].unique())
    selected_countries = st.sidebar.multiselect(
        "Select Countries", 
        available_countries,
        default=[]  # No countries selected by default
    )
    
    # Calculate metrics with fallbacks
    try:
        total_population = pd.to_numeric(population_data['response'], errors='coerce').sum()
        total_population = f"{total_population/1000000:.1f}M" if total_population > 0 else "Data pending"
    except:
        total_population = "Data pending"
    
    # Apply category filter only if categories are selected
    if selected_categories:
        filtered_data = filtered_data[filtered_data['category'].isin(selected_categories)]
    
    # Apply country filter only if countries are selected  
    if selected_countries:
        filtered_data = filtered_data[filtered_data['country'].isin(selected_countries)]
    
    # Regional Demographics and Economic Overview
    st.markdown('<div class="section-header"><h2>🌍 Regional Demographics and Economic Overview</h2></div>', unsafe_allow_html=True)
    
    # Filter data for demographic and economic categories
    demographic_categories = ['Population and Economy', 'Mortality per 100 000 population', 'Mortality per 1000 live births']
    demographic_data = landscape_data[landscape_data['category'].isin(demographic_categories)]
    
    # Calculate demographic metrics
    total_countries = landscape_data['country'].nunique()
    
    # Extract specific indicators for calculations
    population_data = demographic_data[demographic_data['indicator'].str.contains('Population', case=False, na=False)]
    health_expenditure_data = demographic_data[demographic_data['indicator'].str.contains('Total Health Expenditure|Health expenditure|health spending', case=False, na=False)]
    life_expectancy_data = demographic_data[demographic_data['indicator'].str.contains('Life expectancy', case=False, na=False)]
    mortality_100k_data = demographic_data[demographic_data['category'] == 'Mortality per 100 000 population']
    mortality_1000_data = demographic_data[demographic_data['category'] == 'Mortality per 1000 live births']
    
    # Calculate metrics with fallbacks
    try:
        total_population = pd.to_numeric(population_data['response'], errors='coerce').sum()
        total_population = f"{total_population/1000000:.1f}M" if total_population > 0 else "Data pending"
    except:
        total_population = "Data pending"
    
    try:
        health_expenditure = pd.to_numeric(health_expenditure_data['response'], errors='coerce').sum()
        health_expenditure = f"${health_expenditure/1000000:.1f}M" if health_expenditure > 0 else "Data pending"
    except:
        health_expenditure = "Data pending"
    
    try:
        avg_life_expectancy = pd.to_numeric(life_expectancy_data['response'], errors='coerce').mean()
        avg_life_expectancy = f"{avg_life_expectancy:.1f} years" if not pd.isna(avg_life_expectancy) else "Data pending"
    except:
        avg_life_expectancy = "Data pending"
    
    try:
        avg_mortality_100k = pd.to_numeric(mortality_100k_data['response'], errors='coerce').mean()
        avg_mortality_100k = f"{avg_mortality_100k:.1f}/100k" if not pd.isna(avg_mortality_100k) else "Data pending"
    except:
        avg_mortality_100k = "Data pending"
    
    try:
        avg_mortality_1000 = pd.to_numeric(mortality_1000_data['response'], errors='coerce').mean()
        avg_mortality_1000 = f"{avg_mortality_1000:.1f}/1000" if not pd.isna(avg_mortality_1000) else "Data pending"
    except:
        avg_mortality_1000 = "Data pending"
    
    # Display demographic overview in styled cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #0093D5; margin-bottom: 15px;">🌍 Regional Coverage</h3>
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <span style="font-size: 24px; margin-right: 10px;">🏛️</span>
                <div>
                    <strong style="font-size: 18px; color: #003C71;">47 Countries</strong><br>
                    <small style="color: #666;">Surveyed Countries</small>
                </div>
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <span style="font-size: 24px; margin-right: 10px;">👥</span>
                <div>
                    <strong style="font-size: 18px; color: #003C71;">{}</strong><br>
                    <small style="color: #666;">Total Population</small>
                </div>
            </div>
        </div>
        """.format(total_population), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #0093D5; margin-bottom: 15px;">💰 Economic Indicators</h3>
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <span style="font-size: 24px; margin-right: 10px;">🏥</span>
                <div>
                    <strong style="font-size: 18px; color: #003C71;">{}</strong><br>
                    <small style="color: #666;">Health Expenditure</small>
                </div>
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <span style="font-size: 24px; margin-right: 10px;">❤️</span>
                <div>
                    <strong style="font-size: 18px; color: #003C71;">{}</strong><br>
                    <small style="color: #666;">Life Expectancy</small>
                </div>
            </div>
        </div>
        """.format(health_expenditure, avg_life_expectancy), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #0093D5; margin-bottom: 15px;">⚕️ Health Outcomes</h3>
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <span style="font-size: 24px; margin-right: 10px;">📊</span>
                <div>
                    <strong style="font-size: 18px; color: #003C71;">{}</strong><br>
                    <small style="color: #666;">Mortality per 100k pop</small>
                </div>
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <span style="font-size: 24px; margin-right: 10px;">👶</span>
                <div>
                    <strong style="font-size: 18px; color: #003C71;">{}</strong><br>
                    <small style="color: #666;">Mortality per 1000 births</small>
                </div>
            </div>
        </div>
        """.format(avg_mortality_100k, avg_mortality_1000), unsafe_allow_html=True)
    
    # Surveillance (SARI & ILI): Regional Overview
    st.markdown("---")
    st.markdown('<div class="section-header"><h2>🗺️ Surveillance (SARI & ILI): Regional Overview</h2></div>', unsafe_allow_html=True)
    
    
    # Filter data for SARI and ILI surveillance categories
    surveillance_categories = [
        'Severe acute respiratory infection (SARI) surveillance',
        'Influenza like Illness (ILI) Surveillance'
    ]
    surveillance_data = landscape_data[landscape_data['category'].isin(surveillance_categories)]
    
    if not surveillance_data.empty:
        # Create three columns: Map Key on far left, Map in center, Table on right
        col_key, col_map, col_table = st.columns([1, 3, 2])
        
        with col_map:
            st.subheader("🗺️ ILI/SARI Surveillance Implementation Map")
            
            # Prepare data for map visualization
            # Group by country and get key surveillance indicators
            map_data = surveillance_data.groupby(['country', 'category', 'indicator', 'response']).size().reset_index(name='count')
            
            # Create a summary for each country showing their surveillance status
            country_summary = []
            for country in map_data['country'].unique():
                country_data = map_data[map_data['country'] == country]
                
                # Get SARI and ILI status
                sari_data = country_data[country_data['category'] == 'Severe acute respiratory infection (SARI) surveillance']
                ili_data = country_data[country_data['category'] == 'Influenza like Illness (ILI) Surveillance']
                
                # Determine surveillance status based on responses
                sari_status = "Implemented" if len(sari_data) > 0 else "Not Reported"
                ili_status = "Implemented" if len(ili_data) > 0 else "Not Reported"
                
                # Create an overall status
                if sari_status == "Implemented" and ili_status == "Implemented":
                    overall_status = "Both SARI & ILI"
                elif sari_status == "Implemented":
                    overall_status = "SARI Only"
                elif ili_status == "Implemented":
                    overall_status = "ILI Only"
                else:
                    overall_status = "Limited/None"
                
                # Calculate total sentinel sites for this country
                # Get SARI sentinel sites
                sari_sentinel_data = surveillance_data[
                    (surveillance_data['country'] == country) & 
                    (surveillance_data['indicator'] == 'Number of SARI sentinel surveillance sites')
                ]
                sari_sites = 0
                if not sari_sentinel_data.empty:
                    for response in sari_sentinel_data['response']:
                        try:
                            # Handle numeric responses and convert to int
                            if pd.notna(response) and str(response).lower() not in ['no response', 'no', 'none', 'nan']:
                                sari_sites += int(float(str(response)))
                        except (ValueError, TypeError):
                            pass  # Skip non-numeric responses
                
                # Get ILI sentinel sites  
                ili_sentinel_data = surveillance_data[
                    (surveillance_data['country'] == country) & 
                    (surveillance_data['indicator'] == 'Numbers of ILI sentinel surveillance sites')
                ]
                ili_sites = 0
                if not ili_sentinel_data.empty:
                    for response in ili_sentinel_data['response']:
                        try:
                            # Handle numeric responses and convert to int
                            if pd.notna(response) and str(response).lower() not in ['no response', 'no', 'none', 'nan']:
                                ili_sites += int(float(str(response)))
                        except (ValueError, TypeError):
                            pass  # Skip non-numeric responses
                
                total_sentinel_sites = sari_sites + ili_sites
                
                country_summary.append({
                    'country': country,
                    'sari_status': sari_status,
                    'ili_status': ili_status,
                    'overall_status': overall_status,
                    'total_sentinel_sites': total_sentinel_sites,
                    'sari_sites': sari_sites,
                    'ili_sites': ili_sites
                })
            
            country_df = pd.DataFrame(country_summary)
            
            # Create choropleth map using African country data
            # WHO GIS Guidelines: Use clear, accessible colors and provide legend
            color_map = {
                "Both SARI & ILI": "#0093D5",      # WHO Blue
                "SARI Only": "#4CAF50",            # Green  
                "ILI Only": "#FF9800",             # Orange
                "Limited/None": "#E0E0E0"          # Light gray
            }
            
            # Create the map
            fig_map = px.choropleth(
                country_df,
                locations='country',
                locationmode='country names',
                color='overall_status',
                hover_name='country',
                hover_data={
                    'sari_status': True,
                    'ili_status': True,
                    'total_sentinel_sites': True,
                    'sari_sites': True,
                    'ili_sites': True,
                    'overall_status': False
                },
                color_discrete_map=color_map,
                title="WHO AFRO: SARI & ILI Surveillance Implementation Status",
                labels={'overall_status': 'Surveillance Status'}
            )
            
            # Update layout according to WHO GIS Guidelines
            fig_map.update_layout(
                font={"family": "Montserrat", "size": 12},
                title={
                    "font": {"size": 16, "color": "#003C71"},
                    "x": 0.5,
                    "xanchor": 'center'
                },
                geo={
                    'scope': 'africa',
                    'projection_type': 'natural earth',
                    'showframe': False,
                    'showcoastlines': True,
                    'coastlinecolor': "#CCCCCC",
                    'showland': True,
                    'landcolor': '#F5F5F5',
                    'bgcolor': 'white'
                },
                height=500,
                legend={
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": -0.1,
                    "xanchor": "center",
                    "x": 0.5,
                    "bgcolor": "rgba(255,255,255,0.8)",
                    "bordercolor": "#CCCCCC",
                    "borderwidth": 1
                }
            )
            
            # Custom hover template
            fig_map.update_traces(
                hovertemplate='<b>%{hovertext}</b><br>' +
                             'SARI Status: %{customdata[0]}<br>' +
                             'ILI Status: %{customdata[1]}<br>' +
                             'Total Sentinel Sites: %{customdata[2]}<br>' +
                             'SARI Sites: %{customdata[3]}<br>' +
                             'ILI Sites: %{customdata[4]}<br>' +
                             '<extra></extra>'
            )
            
            st.plotly_chart(fig_map, use_container_width=True, key="surveillance_detailed_map")
        
        with col_key:
            # Map Legend and Key Information - Vertical layout on far left
            st.markdown("""
            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #0093D5; margin-top: 60px;">
                <h4 style="color: #003C71; margin-top: 0; margin-bottom: 15px; text-align: center;">🗝️ Map Key</h4>
                <div style="display: flex; flex-direction: column; gap: 12px;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="color: #0093D5; font-size: 16px;">●</span> 
                        <span style="font-size: 12px;">Both SARI & ILI</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="color: #4CAF50; font-size: 16px;">●</span> 
                        <span style="font-size: 12px;">SARI Only</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="color: #FF9800; font-size: 16px;">●</span> 
                        <span style="font-size: 12px;">ILI Only</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="color: #E0E0E0; font-size: 16px;">●</span> 
                        <span style="font-size: 12px;">Limited/No Data</span>
                    </div>
                </div>
                <p style="margin-bottom: 0; font-size: 10px; color: #666; margin-top: 12px; text-align: center;">
                    <i>WHO GIS Guidelines</i>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        with col_table:
            st.subheader("📊 Surveillance Implementation Summary")
            
            # Calculate comprehensive surveillance metrics
            
            # 1. Total Number of Countries with Surveillance (ILI or SARI)
            # Look for "Type of" surveillance indicators
            sari_type_data = surveillance_data[
                (surveillance_data['indicator'] == 'Type of SARI surveillance') &
                (~surveillance_data['response'].str.lower().isin(['n/a', 'no', 'non', 'no response', 'nan']))
            ]
            ili_type_data = surveillance_data[
                (surveillance_data['indicator'] == 'Type of ILI surveillance') &
                (~surveillance_data['response'].str.lower().isin(['n/a', 'no', 'non', 'no response', 'nan']))
            ]
            total_countries_with_surveillance = len(set(sari_type_data['country'].unique()) | set(ili_type_data['country'].unique()))
            
            # 2. Total Number of SARI sites
            sari_sites_data = surveillance_data[
                surveillance_data['indicator'] == 'Number of SARI sentinel surveillance sites'
            ]
            total_sari_sites = 0
            for response in sari_sites_data['response']:
                try:
                    if pd.notna(response) and str(response).lower() not in ['n/a', 'no', 'non', 'no response', 'nan']:
                        total_sari_sites += int(float(str(response)))
                except (ValueError, TypeError):
                    pass
            
            # 3. Total Number of ILI sites
            ili_sites_data = surveillance_data[
                surveillance_data['indicator'] == 'Numbers of ILI sentinel surveillance sites'
            ]
            total_ili_sites = 0
            for response in ili_sites_data['response']:
                try:
                    if pd.notna(response) and str(response).lower() not in ['n/a', 'no', 'non', 'no response', 'nan']:
                        total_ili_sites += int(float(str(response)))
                except (ValueError, TypeError):
                    pass
            
            # 4. Total Yes responses to ILI laboratory confirmation
            ili_lab_confirmation = surveillance_data[
                (surveillance_data['indicator'] == 'Is laboratory confirmation sought for ILI sentinel surveillance') &
                (surveillance_data['response'].str.lower() == 'yes')
            ]
            total_ili_lab_yes = len(ili_lab_confirmation)
            
            # 5. Total countries using case definitions for surveillance
            sari_case_def = surveillance_data[
                (surveillance_data['indicator'] == 'Surveillance case definition used for SARI case ascertainment.') &
                (~surveillance_data['response'].str.lower().isin(['n/a', 'no', 'non', 'no response', 'nan']))
            ]
            ili_case_def = surveillance_data[
                (surveillance_data['indicator'] == 'Surveillance case definition used for ILI case ascertainment.') &
                (~surveillance_data['response'].str.lower().isin(['n/a', 'no', 'non', 'no response', 'nan']))
            ]
            total_countries_case_def = len(set(sari_case_def['country'].unique()) | set(ili_case_def['country'].unique()))
            
            # 6. Countries with integrated surveillance for COVID-19 and Influenza
            integrated_surveillance_data = landscape_data[
                (landscape_data['category_id'] == 11) &
                (landscape_data['indicator'] == 'Has the country integrated influenza and SARS-CoV-2 sentinel surveillance?') &
                (landscape_data['response'].str.lower() == 'yes')
            ]
            total_countries_integrated = len(integrated_surveillance_data)
            
            # 7. Site type breakdown for SARI surveillance
            sari_site_types = surveillance_data[
                (surveillance_data['indicator'] == 'Sentinel sites involved in SARI surveillance.') &
                (~surveillance_data['response'].str.lower().isin(['n/a', 'no', 'non', 'no response', 'nan']))
            ]
            site_type_counts = sari_site_types['response'].value_counts()
            
            # Display key metrics in a more comprehensive format
            st.markdown("**Implementation Overview:**")
            
            # First row of metrics
            metrics_row1_col1, metrics_row1_col2 = st.columns(2)
            with metrics_row1_col1:
                st.metric(
                    "🏥 Countries with Sentinel Surveillance", 
                    total_countries_with_surveillance,
                    help="Countries with ILI or SARI surveillance (excluding N/A, No responses)"
                )
            with metrics_row1_col2:
                st.metric(
                    "📋 Countries Using Case Definitions", 
                    total_countries_case_def,
                    help="Countries using case definitions for ILI or SARI surveillance"
                )
            
            # Second row of metrics
            metrics_row2_col1, metrics_row2_col2 = st.columns(2)
            with metrics_row2_col1:
                st.metric(
                    "🏢 Total SARI Sites", 
                    total_sari_sites,
                    help="Sum of all SARI sentinel surveillance sites"
                )
            with metrics_row2_col2:
                st.metric(
                    "🏥 Total ILI Sites", 
                    total_ili_sites,
                    help="Sum of all ILI sentinel surveillance sites"
                )
            
            # Third row of metrics
            metrics_row3_col1, metrics_row3_col2 = st.columns(2)
            with metrics_row3_col1:
                st.metric(
                    "🔬 ILI Lab Confirmation (Yes)", 
                    total_ili_lab_yes,
                    help="Countries seeking laboratory confirmation for ILI surveillance"
                )
            with metrics_row3_col2:
                st.metric(
                    "🔗 Integrated COVID-19 & Influenza", 
                    total_countries_integrated,
                    help="Countries with integrated influenza and SARS-CoV-2 sentinel surveillance"
                )
            
            st.markdown("---")
            
            # SARI Site Types Breakdown
            if not site_type_counts.empty:
                st.markdown("**🏢 SARI Site Types:**")
                for site_type, count in site_type_counts.head(5).items():
                    st.write(f"• **{site_type}**: {count} countries")
            
    else:
        st.warning("No SARI or ILI surveillance data available in the dataset.")

    # Surveillance Analytics Visualizations
    st.markdown("---")
    
    if not surveillance_data.empty:
        # Create four columns for horizontal chart alignment
        viz_col1, viz_col2, viz_col3, viz_col4 = st.columns(4)
        
        with viz_col1:
            # Sunburst Chart: Surveillance Implementation Overview
            # Prepare hierarchical data for sunburst chart
            sunburst_data = []
            
            # Calculate surveillance type distribution for sunburst chart
            sari_countries = surveillance_data[
                (surveillance_data['category'] == 'Severe acute respiratory infection (SARI) surveillance') &
                (~surveillance_data['response'].str.lower().isin(['n/a', 'no', 'non', 'no response', 'nan']))
            ]['country'].nunique()
            
            ili_countries = surveillance_data[
                (surveillance_data['category'] == 'Influenza like Illness (ILI) Surveillance') &
                (~surveillance_data['response'].str.lower().isin(['n/a', 'no', 'non', 'no response', 'nan']))
            ]['country'].nunique()
            
            # Create hierarchical sunburst data
            sunburst_data = [
                {
                    'category': 'Surveillance',
                    'type': 'SARI',
                    'metric': 'Countries',
                    'value': sari_countries,
                    'path': ['Surveillance', 'SARI', 'Countries']
                },
                {
                    'category': 'Surveillance',
                    'type': 'SARI',
                    'metric': 'Sites',
                    'value': total_sari_sites,
                    'path': ['Surveillance', 'SARI', 'Sites']
                },
                {
                    'category': 'Surveillance',
                    'type': 'ILI',
                    'metric': 'Countries',
                    'value': ili_countries,
                    'path': ['Surveillance', 'ILI', 'Countries']
                },
                {
                    'category': 'Surveillance',
                    'type': 'ILI',
                    'metric': 'Sites',
                    'value': total_ili_sites,
                    'path': ['Surveillance', 'ILI', 'Sites']
                }
            ]
            
            sunburst_df = pd.DataFrame(sunburst_data)
            
            # Create sunburst chart
            fig_sunburst = px.sunburst(
                sunburst_df,
                path=[px.Constant('Surveillance'), 'type', 'metric'],
                values='value',
                title="Surveillance Overview",
                color='type',
                color_discrete_map={
                    'SARI': '#0093D5',  # WHO Blue
                    'ILI': '#4CAF50'    # Green
                }
            )
            
            # Update sunburst chart layout
            fig_sunburst.update_layout(
                font={"family": "Montserrat", "size": 10},
                height=350,
                title={
                    "font": {"size": 12, "color": "#003C71"},
                    "x": 0.5,
                    "xanchor": 'center'
                }
            )
            
            st.plotly_chart(fig_sunburst, use_container_width=True, key="surveillance_sunburst")
        
        with viz_col2:
            # 1. Surveillance Type Distribution
            surveillance_type_data = surveillance_data.groupby('category')['country'].nunique().reset_index()
            surveillance_type_data['category'] = surveillance_type_data['category'].str.replace(
                'Severe acute respiratory infection (SARI) surveillance', 'SARI'
            ).str.replace(
                'Influenza like Illness (ILI) Surveillance', 'ILI'
            )
            
            fig_donut1 = px.pie(
                surveillance_type_data,
                values='country',
                names='category',
                title="Countries by Type",
                color_discrete_sequence=['#0093D5', '#4CAF50', '#FF9800'],
                hole=0.4
            )
            fig_donut1.update_layout(
                font={"family": "Montserrat", "size": 10},
                height=350,
                title={"font": {"size": 12, "color": "#003C71"}},
                showlegend=True,
                legend={"orientation": "h", "yanchor": "bottom", "y": -0.2, "xanchor": "center", "x": 0.5}
            )
            st.plotly_chart(fig_donut1, use_container_width=True, key="surveillance_donut1")
            
        with viz_col3:
            # 2. Lab Confirmation Status
            lab_confirmation_data = pd.DataFrame({
                'status': ['With Lab Confirmation', 'Without Lab Confirmation'],
                'count': [total_ili_lab_yes, max(0, surveillance_data['country'].nunique() - total_ili_lab_yes)]
            })
            
            fig_donut2 = px.pie(
                lab_confirmation_data,
                values='count',
                names='status',
                title="Lab Confirmation",
                color_discrete_sequence=['#0093D5', '#E0E0E0'],
                hole=0.4
            )
            fig_donut2.update_layout(
                font={"family": "Montserrat", "size": 10},
                height=350,
                title={"font": {"size": 12, "color": "#003C71"}},
                showlegend=True,
                legend={"orientation": "h", "yanchor": "bottom", "y": -0.2, "xanchor": "center", "x": 0.5}
            )
            st.plotly_chart(fig_donut2, use_container_width=True, key="surveillance_donut2")
            
        with viz_col4:
            # 3. Case Definition Usage
            case_def_data = pd.DataFrame({
                'status': ['Using Case Definitions', 'Not Using Case Definitions'],
                'count': [total_countries_case_def, max(0, surveillance_data['country'].nunique() - total_countries_case_def)]
            })
            
            fig_donut3 = px.pie(
                case_def_data,
                values='count',
                names='status',
                title="Case Definitions",
                color_discrete_sequence=['#4CAF50', '#E0E0E0'],
                hole=0.4
            )
            fig_donut3.update_layout(
                font={"family": "Montserrat", "size": 10},
                height=350,
                title={"font": {"size": 12, "color": "#003C71"}},
                showlegend=True,
                legend={"orientation": "h", "yanchor": "bottom", "y": -0.2, "xanchor": "center", "x": 0.5}
            )
            st.plotly_chart(fig_donut3, use_container_width=True, key="surveillance_donut3")

    # Vaccination Overview
    st.markdown("---")
    st.markdown('<div class="section-header"><h2>💉 Vaccination Overview</h2></div>', unsafe_allow_html=True)
    
    # Filter data for vaccination category (category_id = 9)
    vaccination_data = landscape_data[landscape_data['category_id'] == 9]
    
    if not vaccination_data.empty:
        # Create three columns: Map Key on far left, Map in center, Table on right
        vax_col_key, vax_col_map, vax_col_table = st.columns([1, 3, 2])
        
        with vax_col_map:
            st.subheader("🗺️ Influenza Vaccination Policy Implementation Map")
            
            # Prepare data for vaccination map visualization
            vax_map_data = vaccination_data.groupby(['country', 'indicator', 'response']).size().reset_index(name='count')
            
            # Create a summary for each country showing their vaccination policy status
            vax_country_summary = []
            for country in vax_map_data['country'].unique():
                country_data = vax_map_data[vax_map_data['country'] == country]
                
                # Get vaccination policy indicators
                formal_policy_data = country_data[
                    country_data['indicator'] == 'Does the country have a formal seasonal influenza vaccination policy?'
                ]
                public_sector_data = country_data[
                    country_data['indicator'] == 'Has seasonal influenza vaccine been introduced in the public sector? Year introduced;'
                ]
                private_sector_data = country_data[
                    country_data['indicator'] == 'Has seasonal influenza vaccine been introduced in the private sector?'
                ]
                
                # Determine policy status
                has_formal_policy = False
                if not formal_policy_data.empty:
                    responses = formal_policy_data['response'].str.lower()
                    has_formal_policy = any(resp in ['yes', 'y'] for resp in responses if pd.notna(resp))
                
                # Determine public sector introduction
                has_public_sector = False
                public_year = "N/A"
                if not public_sector_data.empty:
                    for response in public_sector_data['response']:
                        if pd.notna(response) and str(response).lower() not in ['no', 'n/a', 'no response', 'nan']:
                            has_public_sector = True
                            # Try to extract year information
                            response_str = str(response)
                            if any(char.isdigit() for char in response_str):
                                public_year = response_str
                            break
                
                # Determine private sector introduction
                has_private_sector = False
                if not private_sector_data.empty:
                    responses = private_sector_data['response'].str.lower()
                    has_private_sector = any(resp in ['yes', 'y'] for resp in responses if pd.notna(resp))
                
                # Create overall vaccination implementation status
                if has_formal_policy and (has_public_sector or has_private_sector):
                    if has_public_sector and has_private_sector:
                        overall_status = "Full Implementation"
                    elif has_public_sector:
                        overall_status = "Public Sector Only"
                    else:
                        overall_status = "Private Sector Only"
                elif has_formal_policy:
                    overall_status = "Policy Only"
                elif has_public_sector or has_private_sector:
                    overall_status = "Limited Implementation"
                else:
                    overall_status = "No Implementation"
                
                # Get risk groups information
                risk_groups_data = vaccination_data[
                    (vaccination_data['country'] == country) & 
                    (vaccination_data['indicator'] == 'What are the recommended risk groups for influenza vaccine?')
                ]
                risk_groups = "Not specified"
                if not risk_groups_data.empty and not risk_groups_data['response'].isna().all():
                    risk_groups = "; ".join([str(resp) for resp in risk_groups_data['response'] if pd.notna(resp)])
                
                # Get vaccine formulation
                formulation_data = vaccination_data[
                    (vaccination_data['country'] == country) & 
                    (vaccination_data['indicator'] == 'Which vaccine formulation is used in the country: Northern/Southern hemisphere vaccine?')
                ]
                formulation = "Not specified"
                if not formulation_data.empty and not formulation_data['response'].isna().all():
                    formulation = str(formulation_data['response'].iloc[0])
                
                vax_country_summary.append({
                    'country': country,
                    'formal_policy': "Yes" if has_formal_policy else "No",
                    'public_sector': "Yes" if has_public_sector else "No",
                    'private_sector': "Yes" if has_private_sector else "No",
                    'public_year': public_year,
                    'overall_status': overall_status,
                    'risk_groups': risk_groups[:100] + "..." if len(risk_groups) > 100 else risk_groups,
                    'formulation': formulation
                })
            
            vax_country_df = pd.DataFrame(vax_country_summary)
            
            # Create choropleth map for vaccination implementation
            # WHO GIS Guidelines: Use clear, accessible colors
            vax_color_map = {
                "Full Implementation": "#0093D5",      # WHO Blue - full implementation
                "Public Sector Only": "#4CAF50",       # Green - public sector
                "Private Sector Only": "#FF9800",      # Orange - private sector
                "Policy Only": "#9C27B0",              # Purple - policy without implementation
                "Limited Implementation": "#FFC107",    # Amber - limited implementation
                "No Implementation": "#E0E0E0"          # Light gray - no implementation
            }
            
            # Create the vaccination map
            fig_vax_map = px.choropleth(
                vax_country_df,
                locations='country',
                locationmode='country names',
                color='overall_status',
                hover_name='country',
                hover_data={
                    'formal_policy': True,
                    'public_sector': True,
                    'private_sector': True,
                    'public_year': True,
                    'formulation': True,
                    'overall_status': False
                },
                color_discrete_map=vax_color_map,
                title="WHO AFRO: Influenza Vaccination Policy Implementation Status",
                labels={'overall_status': 'Implementation Status'}
            )
            
            # Update layout according to WHO GIS Guidelines
            fig_vax_map.update_layout(
                font={"family": "Montserrat", "size": 12},
                title={
                    "font": {"size": 16, "color": "#003C71"},
                    "x": 0.5,
                    "xanchor": 'center'
                },
                geo={
                    'scope': 'africa',
                    'projection_type': 'natural earth',
                    'showframe': False,
                    'showcoastlines': True,
                    'coastlinecolor': "#CCCCCC",
                    'showland': True,
                    'landcolor': '#F5F5F5',
                    'bgcolor': 'white'
                },
                height=500,
                legend={
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": -0.1,
                    "xanchor": "center",
                    "x": 0.5,
                    "bgcolor": "rgba(255,255,255,0.8)",
                    "bordercolor": "#CCCCCC",
                    "borderwidth": 1
                }
            )
            
            # Custom hover template for vaccination map
            fig_vax_map.update_traces(
                hovertemplate='<b>%{hovertext}</b><br>' +
                             'Formal Policy: %{customdata[0]}<br>' +
                             'Public Sector: %{customdata[1]}<br>' +
                             'Private Sector: %{customdata[2]}<br>' +
                             'Public Year: %{customdata[3]}<br>' +
                             'Formulation: %{customdata[4]}<br>' +
                             '<extra></extra>'
            )
            
            st.plotly_chart(fig_vax_map, use_container_width=True, key="vaccination_map")
        
        with vax_col_key:
            # Vaccination Map Legend and Key Information - Vertical layout on far left
            st.markdown("""
            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #0093D5; margin-top: 60px;">
                <h4 style="color: #003C71; margin-top: 0; margin-bottom: 15px; text-align: center;">🗝️ Map Key</h4>
                <div style="display: flex; flex-direction: column; gap: 8px;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="color: #0093D5; font-size: 14px;">●</span> 
                        <span style="font-size: 11px;">Full Implementation</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="color: #4CAF50; font-size: 14px;">●</span> 
                        <span style="font-size: 11px;">Public Sector Only</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="color: #FF9800; font-size: 14px;">●</span> 
                        <span style="font-size: 11px;">Private Sector Only</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="color: #9C27B0; font-size: 14px;">●</span> 
                        <span style="font-size: 11px;">Policy Only</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="color: #FFC107; font-size: 14px;">●</span> 
                        <span style="font-size: 11px;">Limited Implementation</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="color: #E0E0E0; font-size: 14px;">●</span> 
                        <span style="font-size: 11px;">No Implementation</span>
                    </div>
                </div>
                <p style="margin-bottom: 0; font-size: 10px; color: #666; margin-top: 12px; text-align: center;">
                    <i>WHO GIS Guidelines</i>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        with vax_col_table:
            st.subheader("📊 Vaccination Implementation Summary")
            
            # Calculate comprehensive vaccination metrics
            
            # 1. Countries with formal vaccination policy
            formal_policy_yes = vaccination_data[
                (vaccination_data['indicator'] == 'Does the country have a formal seasonal influenza vaccination policy?') &
                (vaccination_data['response'].str.lower() == 'yes')
            ]
            total_countries_formal_policy = len(formal_policy_yes)
            
            # 2. Countries with public sector implementation
            public_sector_data = vaccination_data[
                (vaccination_data['indicator'] == 'Has seasonal influenza vaccine been introduced in the public sector? Year introduced;') &
                (~vaccination_data['response'].str.lower().isin(['no', 'n/a', 'no response', 'nan']))
            ]
            total_countries_public = len(public_sector_data)
            
            # 3. Countries with private sector implementation
            private_sector_yes = vaccination_data[
                (vaccination_data['indicator'] == 'Has seasonal influenza vaccine been introduced in the private sector?') &
                (vaccination_data['response'].str.lower() == 'yes')
            ]
            total_countries_private = len(private_sector_yes)
            
            # 4. Countries with hemisphere vaccine formulation (excluding negative responses)
            hemisphere_vaccine_data = vaccination_data[
                (vaccination_data['indicator'] == 'Which vaccine formulation is used in the country: Northern/Southern hemisphere vaccine?') &
                (~vaccination_data['response'].str.lower().isin(['no', 'non', 'no response', 'n/a', 'nan'])) &
                (vaccination_data['response'].notna())
            ]
            total_countries_with_hemisphere_vaccine = len(hemisphere_vaccine_data)
            
            # 5. Countries using Southern hemisphere vaccine (for reference)
            southern_hemisphere = vaccination_data[
                (vaccination_data['indicator'] == 'Which vaccine formulation is used in the country: Northern/Southern hemisphere vaccine?') &
                (vaccination_data['response'].str.contains('Southern', case=False, na=False))
            ]
            total_southern_hemisphere = len(southern_hemisphere)
            
            # 6. Most common risk groups
            risk_groups_data = vaccination_data[
                vaccination_data['indicator'] == 'What are the recommended risk groups for influenza vaccine?'
            ]
            
            # Display key vaccination metrics
            st.markdown("**Implementation Overview:**")
            
            # First row of metrics
            vax_metrics_row1_col1, vax_metrics_row1_col2 = st.columns(2)
            with vax_metrics_row1_col1:
                st.metric(
                    "📋 Countries with Formal Policy", 
                    total_countries_formal_policy,
                    help="Countries with formal seasonal influenza vaccination policy"
                )
            with vax_metrics_row1_col2:
                st.metric(
                    "🏥 Countries with Public Sector", 
                    total_countries_public,
                    help="Countries with vaccine introduced in public sector"
                )
            
            # Second row of metrics
            vax_metrics_row2_col1, vax_metrics_row2_col2 = st.columns(2)
            with vax_metrics_row2_col1:
                st.metric(
                    "🏢 Countries with Private Sector", 
                    total_countries_private,
                    help="Countries with vaccine introduced in private sector"
                )
            with vax_metrics_row2_col2:
                # Calculate comprehensive implementation (either public or private)
                all_implementation = set(public_sector_data['country'].tolist() + private_sector_yes['country'].tolist())
                st.metric(
                    "💉 Countries with Any Implementation", 
                    len(all_implementation),
                    help="Countries with vaccine in public or private sector"
                )
            
            # Third row of metrics
            vax_metrics_row3_col1, vax_metrics_row3_col2 = st.columns(2)
            with vax_metrics_row3_col1:
                st.metric(
                    "🌍 Southern + Northern Hemisphere Vaccine", 
                    total_countries_with_hemisphere_vaccine,
                    help="Countries with specified Northern/Southern hemisphere vaccine formulation (excluding no, non, no response, n/a)"
                )
            with vax_metrics_row3_col2:
                st.metric(
                    "🌍 Southern Hemisphere Vaccine", 
                    total_southern_hemisphere,
                    help="Countries using Southern hemisphere vaccine formulation"
                )
            
            st.markdown("---")
            
            # Risk Groups Analysis
            if not risk_groups_data.empty:
                st.markdown("**🎯 Most Common Risk Groups:**")
                
                # Extract and count risk groups
                all_risk_groups = []
                for response in risk_groups_data['response']:
                    if pd.notna(response) and str(response).lower() not in ['n/a', 'no', 'non', 'no response', 'nan']:
                        # Split by common delimiters and clean
                        groups = str(response).replace(',', ';').replace('and', ';').split(';')
                        for group in groups:
                            clean_group = group.strip()
                            if clean_group and len(clean_group) > 2:
                                all_risk_groups.append(clean_group)
                
                if all_risk_groups:
                    risk_group_counts = pd.Series(all_risk_groups).value_counts()
                    for group, count in risk_group_counts.head(3).items():
                        st.markdown(f"• **{group}**: {count} countries")
                else:
                    st.markdown("• No risk group data available")
            
    else:
        st.warning("No vaccination data available in the dataset.")

    # Laboratory Capacity Overview
    st.markdown("---")
    st.markdown('<div class="section-header"><h2>🔬 Laboratory Capacity Overview</h2></div>', unsafe_allow_html=True)
    
    # Filter data for virological surveillance category (category_id = 6)
    laboratory_data = landscape_data[landscape_data['category_id'] == 6]
    
    if not laboratory_data.empty:
        # Create three columns: Map Key on far left, Map in center, Table on right
        lab_col_key, lab_col_map, lab_col_table = st.columns([1, 3, 2])
        
        with lab_col_map:
            st.subheader("🗺️ Laboratory Capacity Implementation Map")
            
            # Prepare data for laboratory map visualization
            lab_map_data = laboratory_data.groupby(['country', 'indicator', 'response']).size().reset_index(name='count')
            
            # Create a summary for each country showing their laboratory capacity status
            lab_country_summary = []
            for country in lab_map_data['country'].unique():
                country_data = lab_map_data[lab_map_data['country'] == country]
                
                # Get key laboratory indicators
                nic_data = country_data[
                    country_data['indicator'] == 'Does the country have a designated National Influenza Centre (NIC)'
                ]
                ref_lab_data = country_data[
                    country_data['indicator'] == 'Does the country have a designated National Influenza reference laboratory?'
                ]
                rtpcr_data = country_data[
                    country_data['indicator'] == 'Does the country have RT-PCR capacity?'
                ]
                sequencing_data = country_data[
                    country_data['indicator'] == 'Does the country have genomic sequencing capacity?'
                ]
                
                # Determine laboratory capacity status
                has_nic = False
                if not nic_data.empty:
                    responses = nic_data['response'].str.lower()
                    has_nic = any(resp in ['yes', 'y'] for resp in responses if pd.notna(resp))
                
                has_ref_lab = False
                if not ref_lab_data.empty:
                    responses = ref_lab_data['response'].str.lower()
                    has_ref_lab = any(resp in ['yes', 'y'] for resp in responses if pd.notna(resp))
                
                has_rtpcr = False
                if not rtpcr_data.empty:
                    responses = rtpcr_data['response'].str.lower()
                    has_rtpcr = any(resp in ['yes', 'y'] for resp in responses if pd.notna(resp))
                
                has_sequencing = False
                if not sequencing_data.empty:
                    responses = sequencing_data['response'].str.lower()
                    has_sequencing = any(resp in ['yes', 'y'] for resp in responses if pd.notna(resp))
                
                # Create overall laboratory capacity status
                capacity_score = sum([has_nic, has_ref_lab, has_rtpcr, has_sequencing])
                
                if capacity_score == 4:
                    overall_status = "Full Capacity"
                elif capacity_score == 3:
                    overall_status = "High Capacity"
                elif capacity_score == 2:
                    overall_status = "Moderate Capacity"
                elif capacity_score == 1:
                    overall_status = "Basic Capacity"
                else:
                    overall_status = "Limited Capacity"
                
                # Get WHO CC forwarding status
                whocc_data = laboratory_data[
                    (laboratory_data['country'] == country) & 
                    (laboratory_data['indicator'] == 'Does the country forward samples to a WHO CC?')
                ]
                forwards_to_whocc = "No"
                if not whocc_data.empty:
                    responses = whocc_data['response'].str.lower()
                    if any(resp in ['yes', 'y'] for resp in responses if pd.notna(resp)):
                        forwards_to_whocc = "Yes"
                
                # Get sample processing capacity
                samples_data = laboratory_data[
                    (laboratory_data['country'] == country) & 
                    (laboratory_data['indicator'] == 'Average number of virological samples processed per week (2024)')
                ]
                weekly_samples = "Not reported"
                if not samples_data.empty and not samples_data['response'].isna().all():
                    for response in samples_data['response']:
                        if pd.notna(response) and str(response).lower() not in ['no response', 'n/a', 'nan']:
                            try:
                                weekly_samples = str(int(float(str(response))))
                                break
                            except (ValueError, TypeError):
                                weekly_samples = str(response)
                                break
                
                lab_country_summary.append({
                    'country': country,
                    'nic': "Yes" if has_nic else "No",
                    'ref_lab': "Yes" if has_ref_lab else "No",
                    'rtpcr': "Yes" if has_rtpcr else "No",
                    'sequencing': "Yes" if has_sequencing else "No",
                    'overall_status': overall_status,
                    'whocc_forwarding': forwards_to_whocc,
                    'weekly_samples': weekly_samples,
                    'capacity_score': capacity_score
                })
            
            lab_country_df = pd.DataFrame(lab_country_summary)
            
            # Create choropleth map for laboratory capacity
            # WHO GIS Guidelines: Use clear, accessible colors
            lab_color_map = {
                "Full Capacity": "#0093D5",        # WHO Blue - full capacity
                "High Capacity": "#4CAF50",        # Green - high capacity
                "Moderate Capacity": "#FF9800",    # Orange - moderate capacity
                "Basic Capacity": "#FFC107",       # Amber - basic capacity
                "Limited Capacity": "#E0E0E0"      # Light gray - limited capacity
            }
            
            # Create the laboratory capacity map
            fig_lab_map = px.choropleth(
                lab_country_df,
                locations='country',
                locationmode='country names',
                color='overall_status',
                hover_name='country',
                hover_data={
                    'nic': True,
                    'ref_lab': True,
                    'rtpcr': True,
                    'sequencing': True,
                    'whocc_forwarding': True,
                    'weekly_samples': True,
                    'overall_status': False
                },
                color_discrete_map=lab_color_map,
                title="WHO AFRO: Laboratory Capacity Implementation Status",
                labels={'overall_status': 'Capacity Level'}
            )
            
            # Update layout according to WHO GIS Guidelines
            fig_lab_map.update_layout(
                font={"family": "Montserrat", "size": 12},
                title={
                    "font": {"size": 16, "color": "#003C71"},
                    "x": 0.5,
                    "xanchor": 'center'
                },
                geo={
                    'scope': 'africa',
                    'projection_type': 'natural earth',
                    'showframe': False,
                    'showcoastlines': True,
                    'coastlinecolor': "#CCCCCC",
                    'showland': True,
                    'landcolor': '#F5F5F5',
                    'bgcolor': 'white'
                },
                height=500,
                legend={
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": -0.1,
                    "xanchor": "center",
                    "x": 0.5,
                    "bgcolor": "rgba(255,255,255,0.8)",
                    "bordercolor": "#CCCCCC",
                    "borderwidth": 1
                }
            )
            
            # Custom hover template for laboratory map
            fig_lab_map.update_traces(
                hovertemplate='<b>%{hovertext}</b><br>' +
                             'NIC: %{customdata[0]}<br>' +
                             'Reference Lab: %{customdata[1]}<br>' +
                             'RT-PCR: %{customdata[2]}<br>' +
                             'Sequencing: %{customdata[3]}<br>' +
                             'WHO CC Forwarding: %{customdata[4]}<br>' +
                             'Weekly Samples: %{customdata[5]}<br>' +
                             '<extra></extra>'
            )
            
            st.plotly_chart(fig_lab_map, use_container_width=True, key="laboratory_map")
        
        with lab_col_key:
            # Laboratory Map Legend and Key Information - Vertical layout on far left
            st.markdown("""
            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #0093D5; margin-top: 60px;">
                <h4 style="color: #003C71; margin-top: 0; margin-bottom: 15px; text-align: center;">🗝️ Map Key</h4>
                <div style="display: flex; flex-direction: column; gap: 8px;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="color: #0093D5; font-size: 14px;">●</span> 
                        <span style="font-size: 11px;">Full Capacity</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="color: #4CAF50; font-size: 14px;">●</span> 
                        <span style="font-size: 11px;">High Capacity</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="color: #FF9800; font-size: 14px;">●</span> 
                        <span style="font-size: 11px;">Moderate Capacity</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="color: #FFC107; font-size: 14px;">●</span> 
                        <span style="font-size: 11px;">Basic Capacity</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="color: #E0E0E0; font-size: 14px;">●</span> 
                        <span style="font-size: 11px;">Limited Capacity</span>
                    </div>
                </div>
                <p style="margin-bottom: 0; font-size: 10px; color: #666; margin-top: 12px; text-align: center;">
                    <i>WHO GIS Guidelines</i>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        with lab_col_table:
            st.subheader("📊 Laboratory Capacity Summary")
            
            # Calculate comprehensive laboratory metrics
            
            # 1. Countries with National Influenza Centre
            nic_yes = laboratory_data[
                (laboratory_data['indicator'] == 'Does the country have a designated National Influenza Centre (NIC)') &
                (laboratory_data['response'].str.lower() == 'yes')
            ]
            total_countries_nic = len(nic_yes)
            
            # 2. Countries with Reference Laboratory
            ref_lab_yes = laboratory_data[
                (laboratory_data['indicator'] == 'Does the country have a designated National Influenza reference laboratory?') &
                (laboratory_data['response'].str.lower() == 'yes')
            ]
            total_countries_ref_lab = len(ref_lab_yes)
            
            # 3. Countries with RT-PCR capacity
            rtpcr_yes = laboratory_data[
                (laboratory_data['indicator'] == 'Does the country have RT-PCR capacity?') &
                (laboratory_data['response'].str.lower() == 'yes')
            ]
            total_countries_rtpcr = len(rtpcr_yes)
            
            # 4. Countries with sequencing capacity
            sequencing_yes = laboratory_data[
                (laboratory_data['indicator'] == 'Does the country have genomic sequencing capacity?') &
                (laboratory_data['response'].str.lower() == 'yes')
            ]
            total_countries_sequencing = len(sequencing_yes)
            
            # 5. Countries forwarding samples to WHO CC
            whocc_yes = laboratory_data[
                (laboratory_data['indicator'] == 'Does the country forward samples to a WHO CC?') &
                (laboratory_data['response'].str.lower() == 'yes')
            ]
            total_countries_whocc = len(whocc_yes)
            
            # 6. Countries participating in EQAP
            eqap_yes = laboratory_data[
                (laboratory_data['indicator'] == 'Does the laboratory participate in the External Quality Assurance Panel (EQAP)?') &
                (laboratory_data['response'].str.lower() == 'yes')
            ]
            total_countries_eqap = len(eqap_yes)
            
            # Display key laboratory metrics
            st.markdown("**Capacity Overview:**")
            
            # First row of metrics
            lab_metrics_row1_col1, lab_metrics_row1_col2 = st.columns(2)
            with lab_metrics_row1_col1:
                st.metric(
                    "🏛️ Countries with NIC", 
                    total_countries_nic,
                    help="Countries with designated National Influenza Centre"
                )
            with lab_metrics_row1_col2:
                st.metric(
                    "🔬 Countries with Reference Lab", 
                    total_countries_ref_lab,
                    help="Countries with designated National Influenza reference laboratory"
                )
            
            # Second row of metrics
            lab_metrics_row2_col1, lab_metrics_row2_col2 = st.columns(2)
            with lab_metrics_row2_col1:
                st.metric(
                    "🧬 Countries with RT-PCR", 
                    total_countries_rtpcr,
                    help="Countries with RT-PCR capacity"
                )
            with lab_metrics_row2_col2:
                st.metric(
                    "🔍 Countries with Sequencing", 
                    total_countries_sequencing,
                    help="Countries with genomic sequencing capacity"
                )
            
            # Third row of metrics
            lab_metrics_row3_col1, lab_metrics_row3_col2 = st.columns(2)
            with lab_metrics_row3_col1:
                st.metric(
                    "🌍 Countries Forwarding to WHO CC", 
                    total_countries_whocc,
                    help="Countries forwarding samples to WHO Collaborating Centre"
                )
            with lab_metrics_row3_col2:
                st.metric(
                    "✅ Countries in EQAP", 
                    total_countries_eqap,
                    help="Countries participating in External Quality Assurance Panel"
                )
            
            st.markdown("---")
            
            # Calculate average samples processed per week
            samples_data = laboratory_data[
                (laboratory_data['indicator'] == 'Average number of virological samples processed per week (2024)') &
                (~laboratory_data['response'].str.lower().isin(['n/a', 'no', 'no response', 'nan'])) &
                (laboratory_data['response'].notna())
            ]
            
            # Calculate average excluding non-numeric responses
            total_samples = 0
            valid_countries = 0
            for response in samples_data['response']:
                try:
                    sample_count = float(str(response))
                    total_samples += sample_count
                    valid_countries += 1
                except (ValueError, TypeError):
                    pass  # Skip non-numeric responses
            
            avg_samples_per_week = total_samples / valid_countries if valid_countries > 0 else 0
            
            # Fourth row of metrics
            lab_metrics_row4_col1, lab_metrics_row4_col2 = st.columns(2)
            with lab_metrics_row4_col1:
                st.metric(
                    "📊 Ave # of Samples Processed per week", 
                    f"{avg_samples_per_week:.1f}" if avg_samples_per_week > 0 else "No data",
                    help=f"Average weekly virological samples processed across {valid_countries} countries (2024)"
                )
            with lab_metrics_row4_col2:
                st.metric(
                    "🔢 Countries Reporting Sample Data", 
                    valid_countries,
                    help="Countries with valid sample processing data"
                )
            
            st.markdown("---")
            
            # Virological Surveillance Analysis
            lab_types_data = laboratory_data[
                laboratory_data['indicator'] == 'Type of virological surveillance'
            ]
            
            # Get virus isolation capacity data
            virus_isolation_data = laboratory_data[
                (laboratory_data['indicator'] == 'Virus Isolation capacity') &
                (~laboratory_data['response'].str.lower().isin(['no', 'no response', 'n/a', 'nan'])) &
                (laboratory_data['response'].notna())
            ]
            virus_isolation_count = len(virus_isolation_data)
            
            if not lab_types_data.empty:
                st.markdown("**🔬 Virological Surveillance:**")
                
                # Filter surveillance types excluding negative responses and "None"
                filtered_lab_types = lab_types_data[
                    ~lab_types_data['response'].str.lower().isin(['no', 'no response', 'n/a', 'nan', 'none'])
                ]
                lab_type_counts = filtered_lab_types['response'].value_counts()
                
                # Display surveillance types
                for lab_type, count in lab_type_counts.items():
                    if lab_type.lower() == 'sentinel':
                        # Ensure Sentinel shows correct count
                        sentinel_count = len(lab_types_data[
                            (lab_types_data['response'].str.lower() == 'sentinel') |
                            (lab_types_data['response'].str.contains('Sentinel', case=False, na=False))
                        ])
                        st.markdown(f"• **Sentinel**: {sentinel_count} countries")
                    elif lab_type.lower() != 'yes':  # Skip the generic "Yes" response
                        st.markdown(f"• **{lab_type}**: {count} countries")
                
                # Add virus isolation capacity
                st.markdown(f"• **Virus Isolation Capacity**: {virus_isolation_count} countries")
            
    else:
        st.warning("No laboratory capacity data available in the dataset.")

    # Respiratory Pathogens Preparedness Plans
    st.markdown("---")
    st.markdown('<div class="section-header"><h2>🦠 Respiratory Pathogens Preparedness Plans</h2></div>', unsafe_allow_html=True)
    
    # Filter data for pandemic preparedness and response (category_id=10)
    preparedness_data = landscape_data[landscape_data['category_id'] == 10]
    
    if not preparedness_data.empty:
        # Create two columns for metrics and visualizations
        prep_col1, prep_col2 = st.columns([1, 2])
        
        with prep_col1:
            st.markdown("### Key Preparedness Metrics")
            
            # Metric 1: Countries with PRET plans
            pret_data = preparedness_data[
                (preparedness_data['indicator'] == 'Does the country have a respiratory pathogen pandemic preparedness pan (PRET)?') &
                (preparedness_data['response'].str.lower() == 'yes')
            ]
            pret_count = len(pret_data)
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #0093D5; margin: 0;">PRET Plans</h3>
                <h2 style="margin: 0;">{pret_count} countries</h2>
                <p style="margin: 0; color: #666;">with respiratory pathogen preparedness plans</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Metric 2: Countries with other pandemic plans
            other_plans_data = preparedness_data[
                (preparedness_data['indicator'] == 'Does the country have another pandemic preparedness plan available?') &
                (preparedness_data['response'].str.lower() == 'yes')
            ]
            other_plans_count = len(other_plans_data)
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #4CAF50; margin: 0;">Other Pandemic Plans</h3>
                <h2 style="margin: 0;">{other_plans_count} countries</h2>
                <p style="margin: 0; color: #666;">with alternative preparedness plans</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Metric 3: Pandemic simulation exercises
            simulation_data = preparedness_data[
                (preparedness_data['indicator'] == 'Has the country perfomed a influenza simulation exercise(s) for their pandemic preparedness plan?') &
                (preparedness_data['response'].str.lower() == 'yes')
            ]
            simulation_count = len(simulation_data)
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #FF9800; margin: 0;">Simulation Exercises</h3>
                <h2 style="margin: 0;">{simulation_count} countries</h2>
                <p style="margin: 0; color: #666;">conduct pandemic simulations</p>
            </div>
            """, unsafe_allow_html=True)
        
        with prep_col2:
            st.markdown("### Preparedness Plans Implementation")
            
            # Create a map showing preparedness status
            if 'country' in preparedness_data.columns:
                # Aggregate preparedness data by country
                country_prep_summary = []
                
                for country in preparedness_data['country'].unique():
                    country_data = preparedness_data[preparedness_data['country'] == country]
                    
                    # Check for PRET plan
                    has_pret = len(country_data[
                        (country_data['indicator'] == 'Does the country have a respiratory pathogen pandemic preparedness pan (PRET)?') &
                        (country_data['response'].str.lower() == 'yes')
                    ]) > 0
                    
                    # Check for other plans
                    has_other = len(country_data[
                        (country_data['indicator'] == 'Does the country have another pandemic preparedness plan available?') &
                        (country_data['response'].str.lower() == 'yes')
                    ]) > 0
                    
                    # Check for simulations
                    has_simulation = len(country_data[
                        (country_data['indicator'] == 'Has the country perfomed a influenza simulation exercise(s) for their pandemic preparedness plan?') &
                        (country_data['response'].str.lower() == 'yes')
                    ]) > 0
                    
                    # Determine preparedness level
                    if has_pret and has_simulation:
                        prep_level = "Comprehensive"
                        prep_score = 4
                    elif has_pret or (has_other and has_simulation):
                        prep_level = "Good"
                        prep_score = 3
                    elif has_other or has_simulation:
                        prep_level = "Basic"
                        prep_score = 2
                    else:
                        prep_level = "Limited"
                        prep_score = 1
                    
                    country_prep_summary.append({
                        'country': country,
                        'preparedness_level': prep_level,
                        'preparedness_score': prep_score,
                        'has_pret': has_pret,
                        'has_other': has_other,
                        'has_simulation': has_simulation
                    })
                
                prep_summary_df = pd.DataFrame(country_prep_summary)
                
                if not prep_summary_df.empty:
                    # Create choropleth map
                    fig_prep_map = px.choropleth(
                        prep_summary_df,
                        locations='country',
                        locationmode='country names',
                        color='preparedness_score',
                        hover_name='country',
                        hover_data={
                            'preparedness_level': True,
                            'preparedness_score': False,
                            'has_pret': True,
                            'has_other': True,
                            'has_simulation': True
                        },
                        color_continuous_scale=[
                            [0, '#E0E0E0'],    # Limited - Gray
                            [0.33, '#FFC107'], # Basic - Amber
                            [0.66, '#FF9800'], # Good - Orange
                            [1, '#4CAF50']     # Comprehensive - Green
                        ],
                        title="Pandemic Preparedness Status by Country",
                        labels={'preparedness_score': 'Preparedness Level'}
                    )
                    
                    fig_prep_map.update_layout(
                        title_font_size=16,
                        title_x=0.5,
                        geo=dict(
                            scope='africa',
                            showframe=False,
                            showcoastlines=True,
                            projection_type='equirectangular'
                        ),
                        coloraxis_colorbar=dict(
                            title="Preparedness Level",
                            tickvals=[1, 2, 3, 4],
                            ticktext=["Limited", "Basic", "Good", "Comprehensive"]
                        ),
                        height=500
                    )
                    
                    st.plotly_chart(fig_prep_map, use_container_width=True, key="preparedness_map")
                    
                    # Add preparedness summary table
                    st.markdown("### Preparedness Summary")
                    prep_display_df = prep_summary_df[['country', 'preparedness_level', 'has_pret', 'has_other', 'has_simulation']].copy()
                    prep_display_df.columns = ['Country', 'Preparedness Level', 'PRET Plan', 'Other Plans', 'Simulations']
                    prep_display_df = prep_display_df.sort_values('Country')
                    
                    # Style the table
                    st.dataframe(
                        prep_display_df,
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("No preparedness summary data available for mapping.")
            else:
                st.info("Country information not available for preparedness mapping.")
    else:
        st.warning("No pandemic preparedness data available in the dataset.")

    # Respiratory Pathogens Data Reporting and Usage
    st.markdown("---")
    st.markdown('<div class="section-header"><h2>📊 Respiratory Pathogens Data Reporting and Usage</h2></div>', unsafe_allow_html=True)
    
    # Filter data for data reporting & use (category_id=8)
    reporting_data = landscape_data[landscape_data['category_id'] == 8]
    
    if not reporting_data.empty:
        # Create two columns for metrics and visualizations
        report_col1, report_col2 = st.columns([1, 2])
        
        with report_col1:
            st.markdown("### Key Reporting Metrics")
            
            # Metric 1: FluID Reporting
            fluid_data = reporting_data[
                (reporting_data['indicator'] == 'Does the country report to the FluID?') &
                (reporting_data['response'].str.lower() == 'yes')
            ]
            fluid_count = len(fluid_data)
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #0093D5; margin: 0;">FluID Reporting</h3>
                <h2 style="margin: 0;">{fluid_count} countries</h2>
                <p style="margin: 0; color: #666;">report influenza data to FluID</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Metric 2: FluNet Reporting
            flunet_data = reporting_data[
                (reporting_data['indicator'] == 'Does the country report to the FluNet?') &
                (reporting_data['response'].str.lower() == 'yes')
            ]
            flunet_count = len(flunet_data)
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #4CAF50; margin: 0;">FluNet Reporting</h3>
                <h2 style="margin: 0;">{flunet_count} countries</h2>
                <p style="margin: 0; color: #666;">report virological data to FluNet</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Metric 3: Integrated Surveillance Datasets
            integrated_data = reporting_data[
                (reporting_data['indicator'] == 'Are epidemiological and virological sentinel surveillance datasets integrated?') &
                (reporting_data['response'].str.lower() == 'yes')
            ]
            integrated_count = len(integrated_data)
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #FF9800; margin: 0;">Integrated Datasets</h3>
                <h2 style="margin: 0;">{integrated_count} countries</h2>
                <p style="margin: 0; color: #666;">have integrated surveillance datasets</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Metric 4: Burden of Disease Analysis
            burden_data = reporting_data[
                (reporting_data['indicator'] == 'Has the country performed a burden of disease analysis?') &
                (reporting_data['response'].str.lower().isin(['yes', 'in progress']))
            ]
            burden_count = len(burden_data)
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #9C27B0; margin: 0;">Burden Analysis</h3>
                <h2 style="margin: 0;">{burden_count} countries</h2>
                <p style="margin: 0; color: #666;">completed or conducting burden studies</p>
            </div>
            """, unsafe_allow_html=True)
        
        with report_col2:
            st.markdown("### Reporting Compliance Overview")
            
            # Create a map showing reporting compliance
            if 'country' in reporting_data.columns:
                # Aggregate reporting data by country
                country_reporting_summary = []
                
                for country in reporting_data['country'].unique():
                    country_data = reporting_data[reporting_data['country'] == country]
                    
                    # Check for FluID reporting
                    reports_fluid = len(country_data[
                        (country_data['indicator'] == 'Does the country report to the FluID?') &
                        (country_data['response'].str.lower() == 'yes')
                    ]) > 0
                    
                    # Check for FluNet reporting
                    reports_flunet = len(country_data[
                        (country_data['indicator'] == 'Does the country report to the FluNet?') &
                        (country_data['response'].str.lower() == 'yes')
                    ]) > 0
                    
                    # Check for integrated datasets
                    has_integrated = len(country_data[
                        (country_data['indicator'] == 'Are epidemiological and virological sentinel surveillance datasets integrated?') &
                        (country_data['response'].str.lower() == 'yes')
                    ]) > 0
                    
                    # Check for burden analysis
                    has_burden = len(country_data[
                        (country_data['indicator'] == 'Has the country performed a burden of disease analysis?') &
                        (country_data['response'].str.lower().isin(['yes', 'in progress']))
                    ]) > 0
                    
                    # Determine reporting compliance level
                    compliance_score = sum([reports_fluid, reports_flunet, has_integrated, has_burden])
                    
                    if compliance_score >= 3:
                        compliance_level = "High Compliance"
                        compliance_color = 4
                    elif compliance_score == 2:
                        compliance_level = "Moderate Compliance"
                        compliance_color = 3
                    elif compliance_score == 1:
                        compliance_level = "Basic Compliance"
                        compliance_color = 2
                    else:
                        compliance_level = "Limited Compliance"
                        compliance_color = 1
                    
                    country_reporting_summary.append({
                        'country': country,
                        'compliance_level': compliance_level,
                        'compliance_score': compliance_color,
                        'reports_fluid': reports_fluid,
                        'reports_flunet': reports_flunet,
                        'has_integrated': has_integrated,
                        'has_burden': has_burden,
                        'numeric_score': compliance_score
                    })
                
                reporting_summary_df = pd.DataFrame(country_reporting_summary)
                
                if not reporting_summary_df.empty:
                    # Create choropleth map
                    fig_reporting_map = px.choropleth(
                        reporting_summary_df,
                        locations='country',
                        locationmode='country names',
                        color='compliance_score',
                        hover_name='country',
                        hover_data={
                            'compliance_level': True,
                            'compliance_score': False,
                            'reports_fluid': True,
                            'reports_flunet': True,
                            'has_integrated': True,
                            'has_burden': True,
                            'numeric_score': True
                        },
                        color_continuous_scale=[
                            [0, '#E0E0E0'],    # Limited - Gray
                            [0.33, '#FFC107'], # Basic - Amber
                            [0.66, '#FF9800'], # Moderate - Orange
                            [1, '#4CAF50']     # High - Green
                        ],
                        title="Respiratory Pathogen Reporting Compliance by Country",
                        labels={'compliance_score': 'Compliance Level'}
                    )
                    
                    fig_reporting_map.update_layout(
                        title_font_size=16,
                        title_x=0.5,
                        geo={
                            'scope': 'africa',
                            'showframe': False,
                            'showcoastlines': True,
                            'projection_type': 'equirectangular'
                        },
                        coloraxis_colorbar={
                            'title': "Compliance Level",
                            'tickvals': [1, 2, 3, 4],
                            'ticktext': ["Limited", "Basic", "Moderate", "High"]
                        },
                        height=500
                    )
                    
                    st.plotly_chart(fig_reporting_map, use_container_width=True, key="reporting_map")
                    
                    # Add reporting compliance summary
                    st.markdown("### Reporting Compliance Distribution")
                    
                    # Create a summary chart
                    compliance_counts = reporting_summary_df['compliance_level'].value_counts()
                    
                    fig_compliance = px.pie(
                        values=compliance_counts.values,
                        names=compliance_counts.index,
                        title="Distribution of Reporting Compliance Levels",
                        color_discrete_map={
                            'High Compliance': '#4CAF50',
                            'Moderate Compliance': '#FF9800', 
                            'Basic Compliance': '#FFC107',
                            'Limited Compliance': '#E0E0E0'
                        }
                    )
                    
                    fig_compliance.update_layout(
                        title_font_size=14,
                        title_x=0.5,
                        height=400,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig_compliance, use_container_width=True, key="compliance_pie")
                    
                    # Add detailed reporting table
                    st.markdown("### Detailed Reporting Status")
                    reporting_display_df = reporting_summary_df[['country', 'compliance_level', 'reports_fluid', 'reports_flunet', 'has_integrated', 'has_burden']].copy()
                    reporting_display_df.columns = ['Country', 'Compliance Level', 'FluID', 'FluNet', 'Integrated Data', 'Burden Analysis']
                    reporting_display_df = reporting_display_df.sort_values('Country')
                    
                    # Style the table
                    st.dataframe(
                        reporting_display_df,
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("No reporting compliance data available for mapping.")
            else:
                st.info("Country information not available for reporting compliance mapping.")
    else:
        st.warning("No respiratory pathogen reporting data available in the dataset.")

    # Response Analysis by Category and Indicator
    st.markdown("---")
    st.markdown('<div class="section-header"><h2>📋 Response Analysis by Category and Indicator</h2></div>', unsafe_allow_html=True)
    
    # Category filter for the response analysis table
    selected_analysis_category = st.selectbox(
        "Select Category for Response Analysis",
        options=['All Categories'] + sorted(landscape_data['category'].unique()),
        key="response_analysis_category"
    )
    
    # Filter data based on category selection
    if selected_analysis_category == 'All Categories':
        analysis_data = landscape_data.copy()
    else:
        analysis_data = landscape_data[landscape_data['category'] == selected_analysis_category]
    
    if not analysis_data.empty:
        # Group by Category, Indicator, Response and count countries
        response_summary = analysis_data.groupby(['category', 'indicator', 'response']).agg({
            'country': 'count'
        }).reset_index()
        
        # Rename the count column
        response_summary.rename(columns={'country': '# of Countries Responded'}, inplace=True)
        
        # Sort by indicator, then by response as requested
        response_summary = response_summary.sort_values([
            'indicator',
            'response'
        ], ascending=[True, True])
        
        # Add alternating row colors based on indicator changes
        response_summary['row_color'] = ''
        current_indicator = None
        color_index = 0
        
        for idx, row in response_summary.iterrows():
            if current_indicator != row['indicator']:
                current_indicator = row['indicator']
                color_index = (color_index + 1) % 2
            
            if color_index == 0:
                response_summary.at[idx, 'row_color'] = 'background-color: #f8f9fa;'  # Very light gray
            else:
                response_summary.at[idx, 'row_color'] = 'background-color: #ffffff;'  # White
        
        # Display the response analysis table
        st.markdown("**Response Analysis Table:**")
        
        # Add indicator filter dropdown
        available_indicators = ['All Indicators'] + sorted(response_summary['indicator'].unique())
        selected_indicator_filter = st.selectbox(
            "🔍 Select Indicator:",
            available_indicators,
            key="indicator_filter"
        )
        
        # Apply indicator filter
        if selected_indicator_filter != 'All Indicators':
            filtered_summary = response_summary[response_summary['indicator'] == selected_indicator_filter]
        else:
            filtered_summary = response_summary
        
        # Display the filtered table with styling
        def highlight_rows(row):
            return [row['row_color']] * len(row)
        
        # Create a copy for display without the row_color column
        display_summary = filtered_summary.drop(columns=['row_color'], errors='ignore').copy()
        
        # Apply styling if row_color column exists
        if 'row_color' in filtered_summary.columns:
            styled_df = display_summary.style.apply(
                lambda row: [filtered_summary.loc[row.name, 'row_color']] * len(row) 
                if row.name in filtered_summary.index else [''] * len(row), 
                axis=1
            ).set_table_styles([
                {
                    'selector': 'thead th',
                    'props': [
                        ('background-color', '#e3f2fd'),
                        ('color', '#1565c0'),
                        ('font-weight', 'bold'),
                        ('border', '1px solid #bbdefb'),
                        ('padding', '12px 8px'),
                        ('text-align', 'center')
                    ]
                },
                {
                    'selector': 'tbody td',
                    'props': [
                        ('border', '1px solid #e0e0e0'),
                        ('padding', '10px 8px'),
                        ('font-size', '14px')
                    ]
                },
                {
                    'selector': 'table',
                    'props': [
                        ('border-collapse', 'collapse'),
                        ('border-radius', '8px'),
                        ('overflow', 'hidden'),
                        ('box-shadow', '0 2px 8px rgba(0,0,0,0.1)')
                    ]
                }
            ])
            
            # Display styled dataframe
            st.markdown("**Response Analysis Table:**")
            st.dataframe(
                styled_df,
                use_container_width=True,
                height=400,
                column_config={
                    "Category": st.column_config.TextColumn("category", width="medium"),
                    "Indicator": st.column_config.TextColumn("indicator", width="large"),
                    "Response": st.column_config.TextColumn("response", width="medium"),
                    "# of Countries Responded": st.column_config.NumberColumn(
                        "# of Countries",
                        help="Number of countries that provided this response",
                        format="%d"
                    )
                }
            )
        else:
            # Fallback without styling
            st.markdown("**Response Analysis Table:**")
            st.dataframe(
                display_summary,
                use_container_width=True,
                height=400,
                column_config={
                    "Category": st.column_config.TextColumn("category", width="medium"),
                    "Indicator": st.column_config.TextColumn("indicator", width="large"),
                    "Response": st.column_config.TextColumn("response", width="medium"),
                    "# of Countries Responded": st.column_config.NumberColumn(
                        "# of Countries",
                        help="Number of countries that provided this response",
                        format="%d"
                    )
                }
            )
        
        # Download button for the response analysis (exclude styling column)
        download_summary = filtered_summary.drop(columns=['row_color'], errors='ignore')
        csv_response_analysis = download_summary.to_csv(index=False)
        st.download_button(
            label="📥 Download Response Analysis as CSV",
            data=csv_response_analysis,
            file_name=f"response_analysis_{selected_analysis_category.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    else:
        st.warning("No data available for the selected category.")
    
    # Limitations and Notes
    st.markdown("---")
    st.markdown('<div class="section-header"><h2>Sources and References</h2></div>', unsafe_allow_html=True)
    
    st.warning("""
    **Sources and References:**

    1. **Population Data**: 1.https://data.worldbank.org/indicator/SP.POP.TOTL

    2. **List of Countries**: https://data.who.int/countries/

    3. **Life Expectancy**: https://data.worldbank.org/indicator/SP.DYN.LE00.IN?view=chart
               
    4. **Indicator Metadata Registry List**: https://www.who.int/data/gho/indicator-metadata-registry/imr-details/3130
               
    5. **Mortality Rate data**: https://data.who.int/indicators/i/E3CAF2B/2322814.
    """)

if __name__ == "__main__":
    main()