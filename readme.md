# Diamond Price Prediction App

## Description

This project consists of a Flask web application that predicts diamond prices based on user inputs. The application uses a trained Random Forest regression model to provide price estimates based on features such as carat, clarity, dimensions, and color.

The project includes:
- app.py: A Flask web app that serves the prediction interface.
- train.py: A script to train the Random Forest model on the diamond dataset.
- features.py: A script to analyze feature importances from the trained model.

## Prerequisites

Make sure you have the following packages installed:

- Flask
- pandas
- scikit-learn
- joblib
- numpy

You can install them using pip:

pip install -r requirements.txt

## Installation

1. Clone the repository:

   git clone https://github.com/yourusername/diamond-price-prediction.git
   cd diamond-price-prediction

2. Ensure you have a diamonds.csv file in the project directory. The dataset should contain the following columns:
   - carat
   - clarity
   - x
   - y
   - z
   - color
   - price (target variable)

## Usage

### Train the Model

Run the train.py script to train the model and save it for later use:

   python train.py

This script will output the Mean Squared Error and R² score, and save the trained model as model.joblib.

### Analyze Feature Importance

Run the features.py script to analyze and display the most important features affecting diamond prices:

python features.py

### Run the Web Application

After training the model, start the Flask web application:

python app.py

Open your web browser and navigate to http://127.0.0.1:5000/ to access the prediction interface.

## API Endpoint

### POST /predict

- Request Body:

  {
      "carat": <float>,
      "clarity": <string>,
      "X": <float>,
      "Y": <float>,
      "Z": <float>,
      "color": <string>
  }

- Response:

  {
      "predicted_price": <float>
  }

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

For questions or feedback, please reach out to your-email@example.com.
