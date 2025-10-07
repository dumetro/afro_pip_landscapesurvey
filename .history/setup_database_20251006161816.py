"""
Database setup script for WHO AFRO Influenza Landscape Survey
This script creates the necessary tables and sample data for the dashboard
"""

import psycopg2
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sys
from datetime import datetime

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'who_afro_db',
    'user': 'postgres',
    'password': 'your_password_here',  # Update with your password
    'port': '5432'
}

def create_connection():
    """Create database connection"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def create_tables(conn):
    """Create necessary tables for the dashboard"""
    
    cursor = conn.cursor()
    
    # Health indicators table
    health_indicators_sql = """
    CREATE TABLE IF NOT EXISTS health_indicators (
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
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    # Vaccination data table
    vaccination_data_sql = """
    CREATE TABLE IF NOT EXISTS vaccination_data (
        id SERIAL PRIMARY KEY,
        country_name VARCHAR(100) NOT NULL,
        country_code VARCHAR(3),
        survey_year INTEGER,
        mortality_rate DECIMAL(6,3),
        dpt_vaccination_rate DECIMAL(5,2),
        measles_vaccination_rate DECIMAL(5,2),
        polio_vaccination_rate DECIMAL(5,2),
        bcg_vaccination_rate DECIMAL(5,2),
        hepatitis_vaccination_rate DECIMAL(5,2),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    # Influenza specific data table
    influenza_surveillance_sql = """
    CREATE TABLE IF NOT EXISTS influenza_surveillance (
        id SERIAL PRIMARY KEY,
        country_name VARCHAR(100) NOT NULL,
        country_code VARCHAR(3),
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
    """
    
    try:
        cursor.execute(health_indicators_sql)
        cursor.execute(vaccination_data_sql)
        cursor.execute(influenza_surveillance_sql)
        conn.commit()
        print("Tables created successfully!")
    except Exception as e:
        print(f"Error creating tables: {e}")
        conn.rollback()
    finally:
        cursor.close()

def generate_sample_data():
    """Generate sample data for the database"""
    
    # African countries in WHO AFRO region
    countries = [
        ('Algeria', 'DZA'), ('Angola', 'AGO'), ('Benin', 'BEN'), ('Botswana', 'BWA'),
        ('Burkina Faso', 'BFA'), ('Burundi', 'BDI'), ('Cameroon', 'CMR'), ('Cape Verde', 'CPV'),
        ('Central African Republic', 'CAF'), ('Chad', 'TCD'), ('Comoros', 'COM'), ('Congo', 'COG'),
        ('Democratic Republic of Congo', 'COD'), ('Côte d\'Ivoire', 'CIV'), ('Equatorial Guinea', 'GNQ'),
        ('Eritrea', 'ERI'), ('Eswatini', 'SWZ'), ('Ethiopia', 'ETH'), ('Gabon', 'GAB'),
        ('Gambia', 'GMB'), ('Ghana', 'GHA'), ('Guinea', 'GIN'), ('Guinea-Bissau', 'GNB'),
        ('Kenya', 'KEN'), ('Lesotho', 'LSO'), ('Liberia', 'LBR'), ('Madagascar', 'MDG'),
        ('Malawi', 'MWI'), ('Mali', 'MLI'), ('Mauritania', 'MRT'), ('Mauritius', 'MUS'),
        ('Mozambique', 'MOZ'), ('Namibia', 'NAM'), ('Niger', 'NER'), ('Nigeria', 'NGA'),
        ('Rwanda', 'RWA'), ('São Tomé and Príncipe', 'STP'), ('Senegal', 'SEN'),
        ('Seychelles', 'SYC'), ('Sierra Leone', 'SLE'), ('South Africa', 'ZAF'),
        ('South Sudan', 'SSD'), ('Tanzania', 'TZA'), ('Togo', 'TGO'), ('Uganda', 'UGA'),
        ('Zambia', 'ZMB'), ('Zimbabwe', 'ZWE')
    ]
    
    np.random.seed(42)
    
    # Generate health indicators data
    health_data = []
    for country, code in countries:
        for year in range(2018, 2024):
            health_data.append({
                'country_name': country,
                'country_code': code,
                'life_expectancy': np.random.normal(62, 8),
                'infant_mortality_rate': np.random.exponential(35),
                'maternal_mortality_rate': np.random.exponential(300),
                'vaccination_coverage': np.random.normal(70, 20),
                'cancer_screening_rate': np.random.normal(35, 25),
                'tobacco_use_prevalence': np.random.normal(15, 8),
                'survey_year': year
            })
    
    # Generate vaccination data
    vaccination_data = []
    for country, code in countries:
        for year in range(2018, 2024):
            vaccination_data.append({
                'country_name': country,
                'country_code': code,
                'survey_year': year,
                'mortality_rate': np.random.normal(8, 2),
                'dpt_vaccination_rate': np.random.normal(75, 15),
                'measles_vaccination_rate': np.random.normal(70, 18),
                'polio_vaccination_rate': np.random.normal(80, 12),
                'bcg_vaccination_rate': np.random.normal(65, 20),
                'hepatitis_vaccination_rate': np.random.normal(72, 16)
            })
    
    # Generate influenza surveillance data
    influenza_data = []
    for country, code in countries:
        for year in range(2018, 2024):
            influenza_data.append({
                'country_name': country,
                'country_code': code,
                'survey_year': year,
                'surveillance_system_exists': np.random.choice([True, False], p=[0.7, 0.3]),
                'laboratories_count': np.random.poisson(3),
                'sentinel_sites_count': np.random.poisson(8),
                'seasonal_vaccination_policy': np.random.choice([True, False], p=[0.6, 0.4]),
                'pandemic_preparedness_score': np.random.normal(60, 20),
                'influenza_cases_reported': np.random.poisson(1500),
                'hospitalization_rate': np.random.exponential(15)
            })
    
    return pd.DataFrame(health_data), pd.DataFrame(vaccination_data), pd.DataFrame(influenza_data)

def insert_sample_data(conn):
    """Insert sample data into the database"""
    
    engine = create_engine(f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
    
    try:
        health_df, vaccination_df, influenza_df = generate_sample_data()
        
        # Ensure values are within reasonable ranges
        health_df['life_expectancy'] = health_df['life_expectancy'].clip(40, 85)
        health_df['infant_mortality_rate'] = health_df['infant_mortality_rate'].clip(5, 150)
        health_df['maternal_mortality_rate'] = health_df['maternal_mortality_rate'].clip(10, 1500)
        health_df['vaccination_coverage'] = health_df['vaccination_coverage'].clip(0, 100)
        health_df['cancer_screening_rate'] = health_df['cancer_screening_rate'].clip(0, 100)
        health_df['tobacco_use_prevalence'] = health_df['tobacco_use_prevalence'].clip(0, 50)
        
        vaccination_df['dpt_vaccination_rate'] = vaccination_df['dpt_vaccination_rate'].clip(0, 100)
        vaccination_df['measles_vaccination_rate'] = vaccination_df['measles_vaccination_rate'].clip(0, 100)
        vaccination_df['polio_vaccination_rate'] = vaccination_df['polio_vaccination_rate'].clip(0, 100)
        vaccination_df['bcg_vaccination_rate'] = vaccination_df['bcg_vaccination_rate'].clip(0, 100)
        vaccination_df['hepatitis_vaccination_rate'] = vaccination_df['hepatitis_vaccination_rate'].clip(0, 100)
        
        influenza_df['pandemic_preparedness_score'] = influenza_df['pandemic_preparedness_score'].clip(0, 100)
        influenza_df['laboratories_count'] = influenza_df['laboratories_count'].clip(0, 20)
        influenza_df['sentinel_sites_count'] = influenza_df['sentinel_sites_count'].clip(0, 50)
        
        # Insert data
        health_df.to_sql('health_indicators', engine, if_exists='append', index=False)
        vaccination_df.to_sql('vaccination_data', engine, if_exists='append', index=False)
        influenza_df.to_sql('influenza_surveillance', engine, if_exists='append', index=False)
        
        print(f"Sample data inserted successfully!")
        print(f"Health indicators: {len(health_df)} records")
        print(f"Vaccination data: {len(vaccination_df)} records")
        print(f"Influenza surveillance: {len(influenza_df)} records")
        
    except Exception as e:
        print(f"Error inserting sample data: {e}")

def main():
    """Main function to set up the database"""
    
    print("Setting up WHO AFRO Influenza Landscape Survey Database...")
    print("=" * 60)
    
    # Create connection
    conn = create_connection()
    if not conn:
        print("Failed to connect to database. Please check your configuration.")
        sys.exit(1)
    
    try:
        # Create tables
        print("Creating tables...")
        create_tables(conn)
        
        # Insert sample data
        print("Inserting sample data...")
        insert_sample_data(conn)
        
        print("=" * 60)
        print("Database setup completed successfully!")
        print("\nNext steps:")
        print("1. Update the database credentials in your Streamlit secrets")
        print("2. Run the Streamlit app: streamlit run app.py")
        
    except Exception as e:
        print(f"Error during setup: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()