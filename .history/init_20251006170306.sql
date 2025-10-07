-- WHO AFRO Influenza Landscape Survey Database Initialization
-- This script creates the necessary tables for the dashboard

-- Health indicators table
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

-- Vaccination data table
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

-- Influenza surveillance table
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

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_health_indicators_country_year ON health_indicators(country_name, survey_year);
CREATE INDEX IF NOT EXISTS idx_vaccination_data_country_year ON vaccination_data(country_name, survey_year);
CREATE INDEX IF NOT EXISTS idx_influenza_surveillance_country_year ON influenza_surveillance(country_name, survey_year);

-- Insert sample data for demonstration
INSERT INTO health_indicators (country_name, country_code, life_expectancy, infant_mortality_rate, maternal_mortality_rate, vaccination_coverage, cancer_screening_rate, tobacco_use_prevalence, survey_year) VALUES
('Algeria', 'DZA', 76.9, 21.8, 112, 95.2, 45.3, 18.7, 2023),
('Angola', 'AGO', 61.2, 51.6, 241, 78.1, 23.8, 12.4, 2023),
('Benin', 'BEN', 61.8, 58.2, 397, 89.4, 31.2, 8.9, 2023),
('Botswana', 'BWA', 69.6, 30.1, 144, 92.7, 67.8, 19.5, 2023),
('Burkina Faso', 'BFA', 61.6, 54.7, 320, 85.3, 18.9, 6.7, 2023);

INSERT INTO vaccination_data (country_name, country_code, survey_year, mortality_rate, dpt_vaccination_rate, measles_vaccination_rate, polio_vaccination_rate, bcg_vaccination_rate, hepatitis_vaccination_rate) VALUES
('Algeria', 'DZA', 2023, 4.3, 95.2, 85.1, 95.7, 92.3, 94.8),
('Angola', 'AGO', 2023, 7.8, 78.1, 67.4, 81.2, 75.6, 79.3),
('Benin', 'BEN', 2023, 8.1, 89.4, 78.9, 91.1, 84.7, 88.2),
('Botswana', 'BWA', 2023, 6.2, 92.7, 87.3, 93.4, 89.1, 91.8),
('Burkina Faso', 'BFA', 2023, 8.9, 85.3, 74.2, 86.8, 82.1, 84.5);

INSERT INTO influenza_surveillance (country_name, country_code, survey_year, surveillance_system_exists, laboratories_count, sentinel_sites_count, seasonal_vaccination_policy, pandemic_preparedness_score, influenza_cases_reported, hospitalization_rate) VALUES
('Algeria', 'DZA', 2023, true, 5, 12, true, 78.5, 2341, 12.3),
('Angola', 'AGO', 2023, true, 3, 8, false, 45.2, 1876, 18.7),
('Benin', 'BEN', 2023, true, 2, 6, false, 52.1, 934, 15.4),
('Botswana', 'BWA', 2023, true, 4, 9, true, 69.8, 567, 9.2),
('Burkina Faso', 'BFA', 2023, false, 1, 4, false, 38.7, 1245, 22.1);