# PEAKFLOW - Race Time Prediction

## Overview
PEAKFLOW is a web application that predicts race times based on GPX files and weather conditions. It uses a machine learning model trained on real race data to provide accurate predictions.

## Features
- Upload GPX files to analyze race routes
- Predict race times based on:
  - Elevation profile
  - Weather conditions (temperature, humidity, wind)
  - Personal records (optional)
- Interactive visualization of:
  - Elevation profile
  - Predicted pace per kilometer
  - Weather impact
- Download results in PDF format

## Installation
1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the application:
```bash
python app.py
```

## Usage
1. Access the web interface at `http://localhost:5000`
2. Upload a GPX file
3. Optionally enter weather conditions and personal records
4. View predictions and analysis
5. Download results in PDF format

## Technical Details
- Built with Python and Flask
- Uses machine learning for predictions
- Interactive visualizations with Plotly
- PDF generation with WeasyPrint

## License
This project is licensed under the MIT License - see the LICENSE file for details.
