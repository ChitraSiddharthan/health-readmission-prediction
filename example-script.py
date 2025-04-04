#!/usr/bin/env python3
"""
Demo script showing how to use the Hospital Readmission Prediction System.
This script generates synthetic data and demonstrates the full workflow.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to the path so we can import the main module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the prediction system
from readmission_predictor import HospitalReadmissionPredictor


def create_demo_data(n_samples=1000):
    """
    Create synthetic patient data for demonstration purposes.
    
    Parameters:
    -----------
    n_samples : int, default=1000
        Number of patient samples to generate
        
    Returns:
    --------
    pandas.DataFrame
        Synthetic patient dataset
    """
    np.random.seed(42)
    
    # Create synthetic features
    age = np.random.normal(65, 15, n_samples).astype(int)
    age = np.clip(age, 18, 100)
    
    gender = np.random.choice(['M', 'F'], n_samples)
    
    num_medications = np.random.poisson(5, n_samples)
    num_procedures = np.random.poisson(2, n_samples)
    num_diagnoses = np.random.poisson(3, n_samples)
    
    time_in_hospital = np.random.poisson(4, n_samples)
    
    num_lab_procedures = np.random.poisson(10, n_samples)
    
    # Emergency, Urgent, Elective
    admission_type = np.random.choice(['Emergency', 'Urgent', 'Elective'], n_samples, p=[0.5, 0.3, 0.2])
    
    # Categorical health indicators
    diabetes = np.random.choice(['No', 'Type 1', 'Type 2'], n_samples, p=[0.7, 0.1, 0.2])
    heart_disease = np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7])
    hypertension = np.random.choice(['Yes', 'No'], n_samples, p=[0.4, 0.6])
    
    # Create target (readmission)
    # Higher probability of readmission based on certain factors
    readmission_prob = 0.1  # Base probability
    
    # Age factor (older patients more likely to be readmitted)
    age_factor = (age - 50) / 50  # Normalized to range roughly -0.5 to 1.0
    
    # Number of diagnoses factor
    diagnoses_factor = num_diagnoses / 5  # More diagnoses, higher readmission risk
    
    # Emergency admission factor
    emergency_factor = np.where(admission_type == 'Emergency', 0.1, 0)
    
    # Health conditions factor
    health_factor = 0.0
    health_factor += np.where(diabetes != 'No', 0.1, 0)
    health_factor += np.where(heart_disease == 'Yes', 0.15, 0)
    health_factor += np.where(hypertension == 'Yes', 0.05, 0)
    
    # Combine factors
    readmission_prob += (0.05 * age_factor + 0.1 * diagnoses_factor + emergency_factor + health_factor)
    readmission_prob = np.clip(readmission_prob, 0.05, 0.8)
    
    # Generate readmission outcomes
    readmitted = np.random.binomial(1, readmission_prob)
    
    # Create DataFrame
    data = pd.DataFrame({
        'age': age,
        'gender': gender,
        'time_in_hospital': time_in_hospital,
        'num_medications': num_medications,
        'num_procedures': num_procedures,
        'num_diagnoses': num_diagnoses,
        'num_lab_procedures': num_lab_procedures,
        'admission_type': admission_type,
        'diabetes': diabetes,
        'heart_disease': heart_disease,
        'hypertension': hypertension,
        'readmitted': readmitted
    })
    
    return data


def run_demo():
    """
    Run the full demonstration of the Hospital Readmission Prediction System.
    """
    print("=" * 80)
    print("Hospital Readmission Prediction System - Demo")
    print("=" * 80)
    
    # Generate synthetic data
    print("\nGenerating synthetic patient data...")
    data = create_demo_data(2000)
    print(f"Generated dataset with {data.shape[0]} patient records and {data.shape[1]} features.")
    
    # Initialize the readmission predictor
    predictor = HospitalReadmissionPredictor(readmission_threshold_days=30)
    
    # Define features
    categorical_features = ['gender', 'admission_type', 'diabetes', 'heart_disease', 'hypertension']
    numerical_features = ['age', 'time_in_hospital', 'num_medications', 'num_procedures', 
                         'num_diagnoses', 'num_lab_procedures']
    
    # Explore the data
    predictor.explore_data(data)
    
    # Preprocess the data
    X, y = predictor.preprocess_data(
        data, 
        target_column='readmitted',
        categorical_features=categorical_features,
        numerical_features=numerical_features
    )
    
    # Create derived features
    data_enriched = predictor.create_derived_features(data)
    X_enriched, y_enriched = predictor.preprocess_data(
        data_enriched, 
        target_column='readmitted'
    )
    
    # Train models
    models = predictor.train_models(X_enriched, y_enriched)
    
    # Save the best model
    predictor.save_model("readmission_model.pkl")
    print("\nModel saved to readmission_model.pkl")
    
    # Load the model (for demonstration)
    print("\nLoading the model from file...")
    loaded_predictor = HospitalReadmissionPredictor()
    loaded_predictor.load_model("readmission_model.pkl")
    
    # Generate predictions for new patients
    print("\nGenerating predictions for new patients...")
    # In a real scenario, this would be new patient data
    # For demonstration, we'll use a subset of our existing data
    new_patients = data_enriched.sample(50, random_state=42).drop(columns=['readmitted'])
    predictions = loaded_predictor.predict(new_patients)
    
    # Generate risk report
    risk_report = loaded_predictor.generate_risk_report(new_patients, predictions)
    
    # Display high-risk patients
    print("\n=== High-Risk Patients (Top 10) ===")
    high_risk = risk_report[risk_report['risk_level'] == 'High'].head(10)
    
    # Display relevant columns
    display_cols = ['age', 'gender', 'time_in_hospital', 'admission_type', 
                    'num_diagnoses', 'readmission_probability', 'risk_level']
    
    print(high_risk[display_cols])
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
