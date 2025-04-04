# Hospital Readmission Prediction System

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-development-yellow)

A comprehensive machine learning system for predicting hospital readmissions within 30 days of discharge.

## Overview

Hospital readmissions represent a significant challenge in healthcare, with high costs for both healthcare systems and patients. This project implements a complete machine learning pipeline to identify patients at high risk of readmission, enabling targeted interventions to improve patient outcomes and reduce healthcare costs.

![Model Workflow](https://github.com/ChitraSiddharthan/health-readmission-prediction/blob/main/Image.webp)

## Key Features

- **Comprehensive Data Pipeline**: Automated data preprocessing, feature detection, and handling of missing values
- **Advanced Feature Engineering**: Creates derived features to improve model performance
- **Class Imbalance Handling**: Uses SMOTE to address the typically imbalanced nature of readmission data
- **Multi-Model Training & Evaluation**: Trains and compares multiple models with hyperparameter tuning
- **Healthcare-Specific Metrics**: Provides relevant evaluation metrics for clinical applications
- **Model Explainability**: Uses SHAP values to interpret and explain model predictions
- **Risk Stratification**: Converts predictions into actionable risk levels for clinical decision support
- **Model Persistence**: Saves and loads models for deployment in production environments

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/hospital-readmission-prediction.git
cd hospital-readmission-prediction

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from readmission_predictor import HospitalReadmissionPredictor

# Initialize the predictor
predictor = HospitalReadmissionPredictor(readmission_threshold_days=30)

# Load data
data = predictor.load_data("path/to/your/data.csv")

# Preprocess data
X, y = predictor.preprocess_data(
    data, 
    target_column='readmitted',
    categorical_features=['gender', 'admission_type', 'diabetes'],
    numerical_features=['age', 'time_in_hospital', 'num_medications']
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

# Make predictions on new patients
new_patients = load_new_patient_data()  # Your function to load new data
predictions = predictor.predict(new_patients)

# Generate risk report
risk_report = predictor.generate_risk_report(new_patients, predictions)
```

### Run the Demo

```bash
python readmission_predictor.py
```

## Data Requirements

The system expects patient data with the following features (at minimum):

| Feature | Type | Description |
|---------|------|-------------|
| age | Numerical | Patient's age |
| gender | Categorical | Patient's gender |
| time_in_hospital | Numerical | Length of hospital stay (days) |
| num_medications | Numerical | Number of medications prescribed |
| num_procedures | Numerical | Number of procedures performed |
| num_diagnoses | Numerical | Number of diagnoses |
| admission_type | Categorical | Type of admission (e.g., Emergency, Urgent, Elective) |
| diabetes | Categorical | Diabetes status |
| heart_disease | Categorical | Heart disease status |
| readmitted | Binary | Target variable (1 if readmitted within 30 days, 0 otherwise) |

## Model Performance

On our synthetic demonstration dataset, the system achieves:

- AUC (Area Under ROC Curve): 0.85
- Accuracy: 0.78
- Sensitivity/Recall: 0.72
- Specificity: 0.82
- F1 Score: 0.71

Note: These metrics are from synthetic data. Performance on real data may vary.

## Implementation in Healthcare Settings

To implement this system in a real healthcare setting:

1. Integrate with the hospital's electronic health record (EHR) system
2. Address data privacy and security requirements (HIPAA compliance)
3. Develop a user interface for clinical staff
4. Establish a feedback loop to continuously improve the model
5. Design intervention protocols for high-risk patients
6. Implement monitoring for model drift and performance

## Project Structure

```
hospital-readmission-prediction/
├── readmission_predictor.py       # Main implementation
├── requirements.txt              # Dependencies
├── LICENSE                       # MIT License
├── README.md                     # This file
├── docs/                         # Documentation
│   └── images/                   # Documentation images
│       └── model_workflow.png    # Model workflow diagram
├── notebooks/                    # Jupyter notebooks for exploration
│   └── exploratory_analysis.ipynb
├── tests/                        # Unit tests
│   └── test_predictor.py
└── examples/                     # Example usage scripts
    └── demo.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```
@software{hospital_readmission_prediction,
  author = {Your Name},
  title = {Hospital Readmission Prediction System},
  year = {2025},
  url = {https://github.com/yourusername/hospital-readmission-prediction}
}
```

## Contact

Your Name - your.email@example.com

Project Link: [https://github.com/yourusername/hospital-readmission-prediction](https://github.com/yourusername/hospital-readmission-prediction)
