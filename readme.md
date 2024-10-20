# Diamond Price Prediction App

## Description

This project consists of a Flask web application that predicts diamond prices based on user inputs. The application uses a trained Random Forest regression model to provide price estimates based on features such as carat, clarity, dimensions, and color.

The project includes:
- features.py: A script to analyze feature importances from the trained model.
- train.py: A script to train the Random Forest model on the diamond dataset.
- app.py: A Flask web app that serves the prediction interface

## Prerequisites

Make sure you have the following packages installed:

```
pip install -r requirements.txt
```

## Installation

1. Clone the repository:
```
   git clone https://github.com/yourusername/diamond-price-prediction.git
   cd diamond-price-prediction
```
## Usage

### Analyze Feature Importance(optional)

Run the features.py script to analyze and display the most important features affecting diamond prices:
```
python features.py
```
### Train the Model

Run the train.py script to train the model and save it for later use:
```
python train.py
```
This script will output the Mean Squared Error and R² score, and save the trained model as model.joblib.

### Run the Web Application

After training the model, start the Flask web application:
```
python app.py
```
Open your web browser and navigate to http://127.0.0.1:5000/ to access the prediction interface.
