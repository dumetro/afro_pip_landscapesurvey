#!/usr/bin/env python3
"""
Test script to verify secrets.toml is being read correctly
"""
import streamlit as st
import os

print("Testing secrets access...")
print(f"Current working directory: {os.getcwd()}")

# Check if secrets file exists
secrets_path = ".streamlit/secrets.toml"
if os.path.exists(secrets_path):
    print(f"✅ Found secrets file at: {secrets_path}")
else:
    print(f"❌ Secrets file not found at: {secrets_path}")

# Try to access secrets
try:
    print("\n--- Checking Streamlit secrets access ---")
    
    if hasattr(st, 'secrets'):
        print("✅ st.secrets is available")
        
        # Check if database section exists
        if 'database' in st.secrets:
            print("✅ [database] section found in secrets")
            print(f"DB_HOST: {st.secrets.database.get('DB_HOST', 'NOT_FOUND')}")
            print(f"DB_NAME: {st.secrets.database.get('DB_NAME', 'NOT_FOUND')}")
            print(f"DB_USER: {st.secrets.database.get('DB_USER', 'NOT_FOUND')}")
            print(f"DB_PORT: {st.secrets.database.get('DB_PORT', 'NOT_FOUND')}")
        else:
            print("❌ [database] section not found, checking top-level...")
            print(f"DB_HOST: {st.secrets.get('DB_HOST', 'NOT_FOUND')}")
            print(f"DB_NAME: {st.secrets.get('DB_NAME', 'NOT_FOUND')}")
            print(f"DB_USER: {st.secrets.get('DB_USER', 'NOT_FOUND')}")
            print(f"DB_PORT: {st.secrets.get('DB_PORT', 'NOT_FOUND')}")
    else:
        print("❌ st.secrets is not available")
        
except Exception as e:
    print(f"❌ Error accessing secrets: {e}")

print("\n--- Test complete ---")