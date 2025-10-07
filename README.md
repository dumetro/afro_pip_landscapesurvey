# WHO AFRO Influenza Landscape Survey Dashboard

A comprehensive Streamlit dashboard for visualizing health indicators and influenza surveillance data across the WHO African Region.

![WHO AFRO Dashboard](https://img.shields.io/badge/WHO-AFRO%20Dashboard-0093D5)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)

## Features

- 🌍 **Interactive Country Profiles**: Detailed health indicators for African countries
- 📊 **Dynamic Visualizations**: Charts and maps for mortality, vaccination, and nutrition data
- 🔄 **Real-time Data**: Connect to PostgreSQL database for live updates
- 📱 **Responsive Design**: WHO-branded styling that works on all devices
- 📈 **Time Series Analysis**: Track health indicators over time
- 📥 **Data Export**: Download filtered data as CSV files

## Installation

### Prerequisites

- Python 3.8 or higher
- PostgreSQL database (optional - will use sample data if not available)

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/dumetro/afro_pip_landscapesurvey.git
   cd afro_pip_landscapesurvey
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Database Setup (Optional)**
   
   If you want to connect to a PostgreSQL database:
   
   a. Create a PostgreSQL database named `who_afro_db`
   
   b. Update database credentials in `setup_database.py`
   
   c. Run the setup script to create tables and sample data:
   ```bash
   python setup_database.py
   ```

4. **Configure Streamlit Secrets**
   
   Create `.streamlit/secrets.toml` from the template:
   ```bash
   cp secrets_template.toml .streamlit/secrets.toml
   ```
   
   Edit `.streamlit/secrets.toml` with your database credentials:
   ```toml
   [database]
   DB_HOST = "localhost"
   DB_NAME = "who_afro_db"
   DB_USER = "postgres"
   DB_PASSWORD = "your_password"
   DB_PORT = "5432"
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

## Usage

### Dashboard Sections

1. **Key Metrics Summary**: Overview of health indicators
2. **Mortality & Life Expectancy**: Trends and geographical distribution
3. **Vaccination Coverage**: Immunization rates and progress tracking
4. **Nutrition & Food Security**: Health status indicators

### Filters

- **Country Selection**: View data for specific countries or all countries
- **Date Range**: Filter data by time period
- **Apply Filters**: Refresh data with new filter criteria

### Data Export

- Download filtered health indicators as CSV
- Download vaccination time series data as CSV
- Data includes timestamp for version tracking

## Database Schema

The application expects the following PostgreSQL tables:

### health_indicators
```sql
CREATE TABLE health_indicators (
    id SERIAL PRIMARY KEY,
    country_name VARCHAR(100) NOT NULL,
    country_code VARCHAR(3),
    life_expectancy DECIMAL(5,2),
    infant_mortality_rate DECIMAL(6,2),
    maternal_mortality_rate DECIMAL(8,2),
    vaccination_coverage DECIMAL(5,2),
    cancer_screening_rate DECIMAL(5,2),
    tobacco_use_prevalence DECIMAL(5,2),
    survey_year INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### vaccination_data
```sql
CREATE TABLE vaccination_data (
    id SERIAL PRIMARY KEY,
    country_name VARCHAR(100) NOT NULL,
    survey_year INTEGER,
    mortality_rate DECIMAL(6,3),
    dpt_vaccination_rate DECIMAL(5,2),
    measles_vaccination_rate DECIMAL(5,2),
    polio_vaccination_rate DECIMAL(5,2),
    bcg_vaccination_rate DECIMAL(5,2),
    hepatitis_vaccination_rate DECIMAL(5,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### influenza_surveillance
```sql
CREATE TABLE influenza_surveillance (
    id SERIAL PRIMARY KEY,
    country_name VARCHAR(100) NOT NULL,
    survey_year INTEGER,
    surveillance_system_exists BOOLEAN,
    laboratories_count INTEGER,
    sentinel_sites_count INTEGER,
    seasonal_vaccination_policy BOOLEAN,
    pandemic_preparedness_score DECIMAL(5,2),
    influenza_cases_reported INTEGER,
    hospitalization_rate DECIMAL(6,3),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Customization

### Adding New Visualizations

1. Add new data loading functions in `app.py`
2. Create visualization functions using Plotly
3. Add new sections to the main dashboard layout

### Styling

The dashboard uses WHO brand colors:
- WHO Blue: `#0093D5`
- WHO Dark Blue: `#003C71`
- WHO Gray: `#F2F2F2`
- WHO Text: `#6E6E6E`

Update the CSS in the `st.markdown()` section to modify styling.

### Database Queries

Modify the SQL queries in the data loading functions to match your database schema.

## Deployment

### Streamlit Cloud

1. Push your code to GitHub
2. Connect your repository to [Streamlit Cloud](https://streamlit.io/cloud)
3. Add your database credentials to Streamlit Cloud secrets
4. Deploy the application

### Docker

Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions or issues:
- Create an issue in this repository
- Contact the WHO AFRO technical team
- Email: afro-pip-support@who.int

## Acknowledgments

- World Health Organization African Region
- Streamlit team for the excellent framework
- Contributors to the health surveillance data