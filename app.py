from flask import Flask, render_template, request
import numpy as np
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained LSTM model
model = load_model('lstm_model/my_model.h5')  # Ensure this path is correct
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Fetch stock data for the given ticker symbol
def get_stock_data(ticker):
    # Fetch data starting from a reasonable date to ensure enough points
    df = yf.download(ticker, start='2022-01-01', end='2023-10-01')  # Adjust the end date as needed
    if df.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    # Use the 'Close' price for predictions and normalize the data
    data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return data, scaled_data, scaler

# Function to predict the next day using LSTM
def predict_next_day(scaled_data, scaler):
    # Ensure there's enough data for prediction
    if len(scaled_data) < 30:
        raise ValueError("Not enough data to make a prediction. Please choose a different ticker or date range.")

    # Prepare the last 30 days of data
    input_data = scaled_data[-30:].reshape(1, 30, 1)

    # Make prediction
    prediction = model.predict(input_data)

    # Inverse scale the prediction back to original
    return scaler.inverse_transform(prediction.reshape(-1, 1))  # Ensure it's 2D before inverse transformation

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form['ticker']
    try:
        # Fetch stock data and make predictions
        data, scaled_data, scaler = get_stock_data(ticker)

        predictions = []

        # Make predictions for the next 30 days
        for _ in range(30):
            predicted_price = predict_next_day(scaled_data, scaler)[0, 0]
            predictions.append(predicted_price)

            # Update scaled_data with the new prediction for the next iteration
            scaled_data = np.append(scaled_data, [[scaler.transform([[predicted_price]])[0][0]]], axis=0)

        # Prepare days and render predictions page
        days = list(range(1, len(predictions) + 1))

        # Debugging: Print days and predictions to verify their values
        print("Days:", days)
        print("Predictions:", predictions)

        # Convert predictions to a list of floats if necessary
        predictions = [float(price) for price in predictions]  # Convert to float

        return render_template('predictions.html', days=days, prices=predictions)
    
    except Exception as e:
        return render_template('index.html', error=str(e))  # Show error on the homepage if something goes wrong

if __name__ == '__main__':
    app.run(debug=True)
