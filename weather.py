import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim

# Function to get the user's location (latitude and longitude)
def get_user_location():
    geolocator = Nominatim(user_agent="weather_app")
    location = geolocator.geocode("")

    if location:
        return location.latitude, location.longitude
    else:
        print("Error getting user location.")
        return None, None

# Function to get the city based on coordinates
def get_city_from_coordinates(latitude, longitude):
    geolocator = Nominatim(user_agent="weather_app")
    location = geolocator.reverse(f"{latitude}, {longitude}")

    if location and 'address' in location.raw:
        city = location.raw['address'].get('city', '')
        return city
    else:
        print("Error getting city from coordinates.")
        return None

# Function to fetch weather data from OpenWeatherMap
def get_weather_data(api_key, city):
    base_url = "http://api.openweathermap.org/data/2.5/forecast"
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric",  # You can change units as per user choice
    }

    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        print("Error fetching weather data.")
        return None

# Function to extract and preprocess weather data
def extract_features_and_labels(data):
    timestamps = []
    temperatures = []

    for entry in data['list']:
        timestamps.append(entry['dt'])
        temperatures.append(entry['main']['temp'])

    return timestamps, temperatures

# Function to perform linear regression
def perform_linear_regression(timestamps, temperatures):
    X = np.array(timestamps).reshape(-1, 1)
    y = np.array(temperatures)

    model = LinearRegression()
    model.fit(X, y)

    return model

# Main function
def main():
    api_key = "9801a950f89519515ee45d59892eb0a0"

    # Get user's location (latitude and longitude)
    latitude, longitude = get_user_location()

    if latitude is not None and longitude is not None:
        # Get city based on coordinates
        city = get_city_from_coordinates(latitude, longitude)

        if city:
            weather_data = get_weather_data(api_key, city)

            if weather_data:
                timestamps, temperatures = extract_features_and_labels(weather_data)
                model = perform_linear_regression(timestamps, temperatures)

                # Example: Predict temperature for the next hour
                next_hour = max(timestamps) + 3600  # Adding one hour (3600 seconds)
                temperature_prediction = model.predict(np.array([[next_hour]]))

                print(f"Predicted temperature for the next hour in {city}: {temperature_prediction[0]:.2f}°C")

                # Example: Plot the historical data and regression line
                plt.scatter(timestamps, temperatures, label="Historical Data")
                plt.plot(timestamps, model.predict(np.array(timestamps).reshape(-1, 1)), color='red', label="Regression Line")
                plt.xlabel("Timestamp")
                plt.ylabel("Temperature (°C)")
                plt.legend()
                plt.show()
        else:
            print("City not found based on coordinates.")
    else:
        print("Unable to determine your location.")

if __name__ == "__main__":
    main()
