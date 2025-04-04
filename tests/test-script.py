#!/usr/bin/env python3
"""
Unit tests for the Hospital Readmission Prediction System.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add the parent directory to the path so we can import the main module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the prediction system
from readmission_predictor import HospitalReadmissionPredictor


class TestHospitalReadmissionPredictor(unittest.TestCase):
    """Test cases for the HospitalReadmissionPredictor class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data once for all test methods."""
        # Create a small synthetic dataset for testing
        np.random.seed(42)
        n_samples = 200
        
        # Features
        age = np.random.normal(65, 15, n_samples).astype(int)
        gender = np.random.choice(['M', 'F'], n_samples)
        time_in_hospital = np.random.poisson(4, n_samples)
        num_medications = np.random.poisson(5, n_samples)
        num_diagnoses = np.random.poisson(3, n_samples)
        admission_type = np.random.choice(['Emergency', 'Urgent', 'Elective'], n_samples)
        
        # Target (readmission)
        readmission_prob = 0.1 + 0.01 * (age - 50) / 10 + 0.05 * (num_diagnoses / 3)
        readmission_prob = np.clip(readmission_prob, 0.05, 0.8)
        readmitted = np.random.binomial(1, readmission_prob)
        
        # Create DataFrame
        cls.data = pd.DataFrame({
            'age': age,
            'gender': gender,
            'time_in_hospital': time_in_hospital,
            'num_medications': num_medications,
            'num_diagnoses': num_diagnoses,
            'admission_type': admission_type,
            'readmitted': readmitted
        })
        
        # Define feature types
        cls.categorical_features = ['gender', 'admission_type']
        cls.numerical_features = ['age', 'time_in_hospital', 'num_medications', 'num_diagnoses']
        
    def setUp(self):
        """Set up test fixture before each test method."""
        self.predictor = HospitalReadmissionPredictor(readmission_threshold_days=30)
        
    def test_initialization(self):
        """Test the initialization of the predictor."""
        self.assertEqual(self.predictor.readmission_threshold_days, 30)
        self.assertIsNone(self.predictor.best_model)
        self.assertIsNone(self.predictor.best_model_name)
        
    def test_preprocess_data(self):
        """Test data preprocessing."""
        X, y = self.predictor.preprocess_data(
            self.data,
            target_column='readmitted',
            categorical_features=self.categorical_features,
            numerical_features=self.numerical_features
        )
        
        # Check that the target was extracted correctly
        self.assertEqual(len(y), len(self.data))
        
        # Check that the categorical and numerical features were set correctly
        self.assertEqual(set(self.predictor.categorical_features), set(self.categorical_features))
        self.assertEqual(set(self.predictor.numerical_features), set(self.numerical_features))
        
        # Check that the preprocessor was created
        self.assertIsNotNone(self.predictor.preprocessor)
        
    def test_create_derived_features(self):
        """Test feature engineering."""
        data_enriched = self.predictor.create_derived_features(self.data)
        
        # Check that new features were created
        self.assertGreater(data_enriched.shape[1], self.data.shape[1])
        
        # Check that specific expected features exist
        if 'age' in self.data.columns and 'num_medications' in self.data.columns:
            self.assertIn('age_medication_interaction', data_enriched.columns)
            
        if 'time_in_hospital' in self.data.columns:
            self.assertIn('los_category', data_enriched.columns)
            
    def test_handle_class_imbalance(self):
        """Test class imbalance handling."""
        # Create imbalanced dataset (10% positive class)
        X = np.random.random((100, 5))
        y = np.zeros(100)
        y[:10] = 1  # 10% positive class
        
        X_resampled, y_resampled = self.predictor.handle_class_imbalance(X, y)
        
        # Check that classes are more balanced after SMOTE
        positive_count = np.sum(y_resampled == 1)
        negative_count = np.sum(y_resampled == 0)
        
        # SMOTE should increase the minority class
        self.assertGreater(positive_count, 10)
        
        # The ratio should be less imbalanced
        original_ratio = 10 / 90  # 0.11
        new_ratio = positive_count / negative_count
        self.assertGreater(new_ratio, original_ratio)
        
    def test_model_training_and_evaluation(self):
        """Test model training and evaluation."""
        # Preprocess the data
        X, y = self.predictor.preprocess_data(
            self.data,
            target_column='readmitted',
            categorical_features=self.categorical_features,
            numerical_features=self.numerical_features
        )
        
        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            self.data.drop('readmitted', axis=1), 
            self.data['readmitted'], 
            test_size=0.2, 
            random_state=42
        )
        
        # Use a small sample and fewer iterations for faster testing
        # This is just to check if the functions run without errors
        self.predictor.models = {
            'logistic_regression': self.predictor.models['logistic_regression']
        }
        
        # Train only logistic regression for speed
        X_small = X_train.sample(min(50, len(X_train)), random_state=42)
        y_small = y_train.loc[X_small.index]
        
        # Simplified training just for testing functionality
        models = self.predictor.train_models(X_small, y_small, cv=2)
        
        # Check that a model was trained
        self.assertIsNotNone(self.predictor.best_model)
        self.assertIsNotNone(self.predictor.best_model_name)
        
    def test_save_and_load_model(self):
        """Test model saving and loading."""
        # Create a simple model for testing
        from sklearn.linear_model import LogisticRegression
        self.predictor.best_model = LogisticRegression()
        self.predictor.best_model_name = 'logistic_regression'
        self.predictor.preprocessor = None  # Simplified for testing
        self.predictor.numerical_features = self.numerical_features
        self.predictor.categorical_features = self.categorical_features
        self.predictor.feature_names = self.numerical_features + self.categorical_features
        self.predictor.target = 'readmitted'
        
        # Save the model
        model_path = "test_model.pkl"
        self.predictor.save_model(model_path)
        
        # Check that the file was created
        self.assertTrue(os.path.exists(model_path))
        
        # Create a new predictor and load the model
        new_predictor = HospitalReadmissionPredictor()
        new_predictor.load_model(model_path)
        
        # Check that the model was loaded correctly
        self.assertEqual(new_predictor.best_model_name, 'logistic_regression')
        self.assertEqual(new_predictor.target, 'readmitted')
        
        # Clean up
        if os.path.exists(model_path):
            os.remove(model_path)
            
    def test_generate_risk_report(self):
        """Test risk report generation."""
        # Create synthetic predictions
        data_sample = self.data.sample(20, random_state=42)
        predictions = np.random.random(20)  # Random probabilities between 0 and 1
        
        # Generate risk report
        risk_report = self.predictor.generate_risk_report(data_sample, predictions)
        
        # Check that the report has the expected columns
        self.assertIn('readmission_probability', risk_report.columns)
        self.assertIn('risk_level', risk_report.columns)
        
        # Check that risk levels were assigned correctly
        self.assertTrue(all(risk_report[risk_report['readmission_probability'] > 0.6]['risk_level'] == 'High'))
        self.assertTrue(all(risk_report[risk_report['readmission_probability'] < 0.3]['risk_level'] == 'Low'))


if __name__ == '__main__':
    unittest.main()
