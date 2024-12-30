import streamlit as st
import requests
import pandas as pd
import folium
from streamlit_folium import st_folium
from joblib import load
import datetime
#import os
from geopy.distance import geodesic
import uuid
import pandas as np
import joblib

# Constants
#API_KEY = os.getenv("WEATHER_API_KEY")  # Use environment variable for API key
API_KEY = "231771a51df34a8db2952122242612"
OPENWEATHERMAP_API_KEY = "9eb33c79be718292d16ec0c6bb82de86"
#OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY") 
OPENCAGE_API_KEY = "53a91cfacb1b486e9e14cd965fb4dcb5"  # Replace with your OpenCage API key


import torch
import torch.nn as nn

# Define the DNN model class (same as during training)
class DNN(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        super(DNN, self).__init__()
        layers = []
        in_features = input_size
        for hidden_units in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            in_features = hidden_units
        layers.append(nn.Linear(in_features, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
pipeline_cls = joblib.load("preprocessing_pipeline_cls_last.pkl")
best_params = joblib.load("dnn_best_params.pkl")

# Initialize the model with best parameters
model = DNN(
    input_size=best_params["input_size"],
    hidden_layers=best_params["hidden_layers"],
    output_size=best_params["output_size"],
)

# Load the model weights
model.load_state_dict(torch.load("dnn_model_last.pth"))
model.eval()  # Set model to evaluation mode

def predict(input_data):
    # Preprocess the input data using the saved pipeline
    preprocessed_data = pipeline_cls.transform(input_data)
    input_tensor = torch.tensor(preprocessed_data, dtype=torch.float32)
    
    # Perform prediction
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_class = torch.max(output, 1)  # Get the class with the highest score
    
    return predicted_class.numpy()

# State and coordinates mapping
def get_state_boundaries():
    return {
        "Alaska": [64.2008, -149.4937, 4], "Alabama": [32.3182, -86.9023, 6],
        "Arkansas": [34.7465, -92.2896, 6], "Arizona": [34.0489, -111.0937, 6],
        "California": [36.7783, -119.4179, 5], "Colorado": [39.5501, -105.7821, 6],
        "Connecticut": [41.6032, -73.0877, 7], "Delaware": [38.9108, -75.5277, 7],
        "Florida": [27.9944, -81.7603, 6], "Georgia": [32.1656, -82.9001, 6],
        "Hawaii": [19.8968, -155.5828, 6], "Iowa": [41.8780, -93.0977, 6],
        "Idaho": [44.0682, -114.7420, 6], "Illinois": [40.6331, -89.3985, 6],
        "Indiana": [40.2672, -86.1349, 6], "Kansas": [39.0119, -98.4842, 6],
        "Kentucky": [37.8393, -84.2700, 6], "Louisiana": [31.2448, -92.1450, 6],
        "Massachusetts": [42.4072, -71.3824, 7], "Maryland": [39.0458, -76.6413, 6],
        "Maine": [45.2538, -69.4455, 6], "Michigan": [44.3148, -85.6024, 6],
        "Minnesota": [46.7296, -94.6859, 6], "Missouri": [37.9643, -91.8318, 6],
        "Mississippi": [32.3547, -89.3985, 6], "Montana": [46.8797, -110.3626, 6],
        "North Carolina": [35.7596, -79.0193, 6], "North Dakota": [47.5515, -101.0020, 6],
        "Nebraska": [41.4925, -99.9018, 6], "New Hampshire": [43.1939, -71.5724, 6],
        "New Jersey": [40.0583, -74.4057, 6], "New Mexico": [34.5199, -105.8701, 6],
        "Nevada": [38.8026, -116.4194, 6], "New York": [40.7128, -74.0060, 6],
        "Ohio": [40.4173, -82.9071, 6], "Oklahoma": [35.0078, -97.0929, 6],
        "Oregon": [43.8041, -120.5542, 6], "Pennsylvania": [41.2033, -77.1945, 6],
        "Puerto Rico": [18.2208, -66.5901, 6], "Rhode Island": [41.5801, -71.4774, 7],
        "South Carolina": [33.8361, -81.1637, 6], "South Dakota": [43.9695, -99.9018, 6],
        "Tennessee": [35.5175, -86.5804, 6], "Texas": [31.9686, -99.9018, 5],
        "Utah": [39.3210, -111.0937, 6], "Virginia": [37.4316, -78.6569, 6],
        "Vermont": [44.5588, -72.5778, 6], "Washington": [47.7511, -120.7401, 6],
        "Wisconsin": [43.7844, -88.7879, 6], "West Virginia": [38.5976, -80.4549, 6],
        "Wyoming": [43.0759, -107.2903, 6]
    }

def magnitude_to_size_class(magnitude):
    """
    Convert wildfire magnitude to size class.
    - Magnitude <= 0.1: Class B
    - 0.1 < Magnitude <= 1: Class C
    - 1 < Magnitude <= 3: Class D
    - 3 < Magnitude <= 10: Class E
    - 10 < Magnitude <= 50: Class F
    - Magnitude > 50: Class G
    """
    if magnitude <= 0.1:
        return 'B'
    elif magnitude <= 1:
        return 'C'
    elif magnitude <= 3:
        return 'D'
    elif magnitude <= 10:
        return 'E'
    elif magnitude <= 50:
        return 'F'
    else:
        return 'G'


# # Helper function to map wind categories to wind speed
# def map_wind_speed(wind_category):
#     wind_mapping = {
#         "Calm": 0,
#         "Breezy": 1,
#         "Windy": 2,
#         "Stormy": 3,
#     }
#     return wind_mapping.get(wind_category, 0)

# Helper function to map state abbreviations to numerical values
state_mapping = {
    'AK': 1, 'AL': 2, 'AR': 3, 'AZ': 4, 'CA': 5, 'CO': 6, 'CT': 7, 'DE': 8, 'FL': 9, 'GA': 10,
    'HI': 11, 'IA': 12, 'ID': 13, 'IL': 14, 'IN': 15, 'KS': 16, 'KY': 17, 'LA': 18, 'MA': 19, 'MD': 20,
    'ME': 21, 'MI': 22, 'MN': 23, 'MO': 24, 'MS': 25, 'MT': 26, 'NC': 27, 'ND': 28, 'NE': 29, 'NH': 30,
    'NJ': 31, 'NM': 32, 'NV': 33, 'NY': 34, 'OH': 35, 'OK': 36, 'OR': 37, 'PA': 38, 'PR': 39, 'RI': 40,
    'SC': 41, 'SD': 42, 'TN': 43, 'TX': 44, 'UT': 45, 'VA': 46, 'VT': 47, 'WA': 48, 'WI': 49, 'WV': 50, 'WY': 51
}

# Mapping from state names to abbreviations
state_name_to_abbreviation = {
    "Alaska": "AK", "Alabama": "AL", "Arkansas": "AR", "Arizona": "AZ", "California": "CA", "Colorado": "CO",
    "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Iowa": "IA",
    "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
    "Massachusetts": "MA", "Maryland": "MD", "Maine": "ME", "Michigan": "MI", "Minnesota": "MN", "Missouri": "MO",
    "Mississippi": "MS", "Montana": "MT", "North Carolina": "NC", "North Dakota": "ND", "Nebraska": "NE",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "Nevada": "NV", "New York": "NY", "Ohio": "OH",
    "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA", "Puerto Rico": "PR", "Rhode Island": "RI",
    "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Virginia": "VA",
    "Vermont": "VT", "Washington": "WA", "Wisconsin": "WI", "West Virginia": "WV", "Wyoming": "WY"
}

def get_vegetation_mapping():
    return {
        "Temperate Evergreen Needleleaf Forest TmpENF": 4,
        "C3 Grassland/Steppe": 9,
        "Open Shrubland": 12,
        "Desert": 14,
        "Polar Desert/Rock/Ice": 15,
        "Secondary Tropical Evergreen Broadleaf Forest": 16,
        "Unknown": 0,
    }

# Cause mapping including 'Unknown'
cause_mapping = {
    'Arson': 1,
    'Campfire': 2,
    'Children': 3,
    'Debris Burning': 4,
    'Equipment Use': 5,
    'Fireworks': 6,
    'Lightning': 7,
    'Miscellaneous': 8,
    'Powerline': 10,
    'Railroad': 11,
    'Smoking': 12,
    'Structure': 13,
    'Missing/Undefined': 9,
}

# Tooltip references for temperature, humidity, and precipitation
temp_tooltip = "Cold: -50 to 0 °C\nCool: 1 to 15 °C\nModerate: 16 to 25 °C\nWarm: 26 to 35 °C"
wind_tooltip = "Calm: 0 m/s\nBreezy: 1-5 m/s\nWindy: 6-10 m/s\nStormy: 11+ m/s"
hum_tooltip = "Very Dry: 0 to 20%\nDry: 21 to 40%\nModerate: 41 to 60%\nHumid: 61 to 80%\nVery Humid: 81 to 100%"
prec_tooltip = "No Precipitation: 0 mm\nLight: 1 to 10 mm\nModerate: 11 to 50 mm\nHeavy: 51 to 100 mm\nVery Heavy: 101 to 200 mm\nExtreme: 201 to 2000 mm"


# Fetch current weather data using WeatherAPI
def fetch_weather(lat, lon):
    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={lat},{lon}"
    try:
        response = requests.get(url).json()
        if "current" in response:
            return {
                "temperature": response["current"]["temp_c"],
                "humidity": response["current"]["humidity"],
                "wind_speed": response["current"]["wind_kph"], 
                "precipitation": response["current"]["precip_mm"],
            }
    except Exception as e:
        st.error(f"Error fetching weather data: {e}")
    return None

# Fetch historical weather data using WeatherAPI
def fetch_historical_weather(lat, lon, date):
    url = f"http://api.weatherapi.com/v1/history.json?key={API_KEY}&q={lat},{lon}&dt={date}"
    try:
        response = requests.get(url).json()
        if "forecast" in response and "forecastday" in response["forecast"]:
            day_data = response["forecast"]["forecastday"][0]["day"]
            return {
                "temperature": day_data["avgtemp_c"],
                "humidity": day_data["avghumidity"],
                "wind_speed": day_data["maxwind_kph"],
                "precipitation": day_data["totalprecip_mm"],
            }
    except Exception as e:
        st.error(f"Error fetching historical weather data: {e}")
    return None


# Fetch nearest weather station using OpenWeatherMap API
def fetch_nearest_station(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/find?lat={lat}&lon={lon}&cnt=1&appid={OPENWEATHERMAP_API_KEY}"
    try:
        response = requests.get(url).json()
        if "list" in response and len(response["list"]) > 0:
            station = response["list"][0]
            station_lat = station["coord"]["lat"]
            station_lon = station["coord"]["lon"]
            return station_lat, station_lon
    except Exception as e:
        st.error(f"Error fetching nearest weather station: {e}")
    return None, None


# Fetch nearest city using OpenCage Geocoding API
def fetch_nearest_city(lat, lon):
    url = f"https://api.opencagedata.com/geocode/v1/json?q={lat}+{lon}&key={OPENCAGE_API_KEY}&limit=1"
    try:
        response = requests.get(url).json()
        if "results" in response and response["results"]:
            city_data = response["results"][0]
            city = city_data["components"].get("city")
            return city, city_data["geometry"]["lat"], city_data["geometry"]["lng"]
    except Exception as e:
        st.error(f"Error fetching nearest city: {e}")
    return None, None, None

# Calculate remoteness as the distance to the nearest city
def calculate_remoteness(lat, lon):
    city, city_lat, city_lon = fetch_nearest_city(lat, lon)
    if city_lat and city_lon:
        return geodesic((lat, lon), (city_lat, city_lon)).km
    return -1  # Default value if calculation fails


# Calculate distance to the nearest weather station
def calculate_distance_to_station(lat, lon):
    station_lat, station_lon = fetch_nearest_station(lat, lon)
    if station_lat is not None and station_lon is not None:
        return geodesic((lat, lon), (station_lat, station_lon)).km
    else:
        st.error("Could not calculate distance to the nearest weather station due to missing station coordinates.")
    return None


import json

# Load thresholds
with open('gmm_thresholds.json', 'r') as f:
    thresholds = json.load(f)

def classify_slope(value, thresholds):
    if value < thresholds['threshold_1']:
        return 0
    elif value < thresholds['threshold_2']:
        return 1
    else:
        return 2


today = datetime.date.today()
current_year, current_month, current_day = today.year, today.month, today.day


# Load the model and preprocessing pipeline
@st.cache(allow_output_mutation=True)
  
def load_model_and_pipeline():
    # Load preprocessing pipeline and best parameters
    pipeline_cls = joblib.load("preprocessing_pipeline_cls_last.pkl")
    best_params = joblib.load("dnn_best_params.pkl")
    
    # Initialize the model
    model = DNN(
        input_size=best_params["input_size"],
        hidden_layers=best_params["hidden_layers"],
        output_size=best_params["output_size"],
    )
    model.load_state_dict(torch.load("dnn_model_last.pth"))
    model.eval()  # Set model to evaluation mode
    return model, pipeline_cls

def load_model(model_name):
        model_paths = {
            # "DNN Model": "DNN_pipeline_cls_mh.pkl",
            "GMM Bayesian Model": "gmm_bayesian_with_pipeline.pkl",
            "MLP Model": "mlp_random_model_with_pipeline.pkl",
        }
        model_path = model_paths.get(model_name)
        if model_path:
            try:
                # Load the model and pipeline together
                model_with_pipeline = load(model_path)
                # If the loaded object is a tuple (model, pipeline), unpack it
                if isinstance(model_with_pipeline, tuple):
                    model, pipeline = model_with_pipeline
                else:
                    # Handle case where only the model is loaded
                    model, pipeline = model_with_pipeline, None
                return model, pipeline
            except Exception as e:
                st.error(f"Failed to load model: {e}")
                return None, None
        else:
            return None, None
    


    
#model, pipeline_cls = load_model()

# App Title
st.title("🌲 Wildfire Size Prediction System")

# Initialize Streamlit app
st.sidebar.title("Settings")
# Sidebar dropdown to select the model
selected_model = st.sidebar.selectbox(
    "Choose a Model",
    options=["DNN Model", "GMM Bayesian Model", "MLP Model"],  # Include DNN Model
    index=0  # Default to DNN Model
)


import copy
import traceback

loaded_model = load_model(selected_model)

if selected_model == "DNN Model":
    # Load the model and pipeline once
    model, pipeline_cls = load_model_and_pipeline()

else:
    model, pipeline_cls = copy.deepcopy(load_model(selected_model))

if model and pipeline_cls:
    st.sidebar.success(f"Loaded {selected_model}")
else:
    st.sidebar.error("Failed to load the selected model.")

# Use tabs for organizing current and historical weather data
tab1, tab2, tab3, tab4 = st.tabs([ "Initial Setup", "Current Weather", "Historical Weather", "Prediction"])
with tab1:
    # State Selection
    state_boundaries = get_state_boundaries()
    state = st.selectbox("Select State:", list(state_boundaries.keys()))
    state_abbreviation = state_name_to_abbreviation[state]  # Get state abbreviation
    state_code = state_mapping[state_abbreviation]  # Get numerical representation
    state_center = state_boundaries[state]

    # Vegetation Type
    vegetation_mapping = get_vegetation_mapping()
    vegetation = st.selectbox("Select Vegetation Type:", list(vegetation_mapping.keys()))
    vegetation_code = vegetation_mapping[vegetation]


    # Streamlit user input for Cause
    cause = st.selectbox("Cause:", list(cause_mapping.keys()))  # Include 'Unknown' as an option
    cause_code = cause_mapping[cause]  # Convert cause to a numerical code

    # Location Selection (Map)
    st.write("Click on the map to select a location within the selected state:")
    m = folium.Map(location=[state_center[0], state_center[1]], zoom_start=state_center[2])
    folium.Marker(location=[state_center[0], state_center[1]], draggable=True).add_to(m)
    map_data = st_folium(m, key="map")

    latitude, longitude = state_center[0], state_center[1]  # Default to state center
    if map_data and map_data.get("last_clicked"):
        last_clicked = map_data.get("last_clicked")
        if last_clicked and "lat" in last_clicked and "lng" in last_clicked:
            latitude, longitude = last_clicked["lat"], last_clicked["lng"]

    st.write(f"Selected Coordinates: Latitude : {latitude}, Longitude : {longitude}")


    # Optional Inputs with Skip Option
    def input_with_skip(label, min_value=None, max_value=None, default_value=None, options=None, tooltip=None, key=None):
        # Ensure the key is unique
        unique_key = key if key else label.replace(" ", "_").lower()

        # Add a checkbox to allow skipping the input
        skip = st.checkbox(f"Skip", key=f"skip_{unique_key}")
        if skip:
            return None  # Return None to indicate the input was skipped

        # Display tooltip if provided
        if tooltip:
            st.markdown(f"{label} <span style='color:blue;' title='{tooltip}'>?</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"{label}", unsafe_allow_html=True)

        # Render the appropriate input widget based on the options
        if options:
            return st.selectbox(label, options, index=options.index(default_value) if default_value in options else 0, key=unique_key)

        return st.slider(label, min_value=min_value, max_value=max_value, value=default_value, key=unique_key)

with tab2:
    st.header("Current Weather Data")
    # Current Weather Data
    fetch_current_weather = st.checkbox("Fetch Current Weather from API", value=True)
    if fetch_current_weather:
        if st.button("Fetch Current Weather", key="fetch_current_weather"):
            weather_data = fetch_weather(latitude, longitude)
            if weather_data:
                st.success("Weather data fetched successfully!")
                st.write(f"Temperature: {weather_data['temperature']} °C")
                st.write(f"Humidity: {weather_data['humidity']}%")
                st.write(f"Wind Speed: {weather_data['wind_speed']} m/s")
                st.write(f"Precipitation: {weather_data['precipitation']} mm")
            else:
                st.error("Could not fetch weather data. Try again later.")
    else:
        temp_current = input_with_skip("Current Temperature (°C):", -50, 50, 20, tooltip=temp_tooltip, key="temp_current")
        wind_current = input_with_skip("Current Wind Speed (m/s):", 0, 25, 5, tooltip=wind_tooltip, key="wind_current")
        hum_current = input_with_skip("Current Humidity (%):", 0, 100, 50, tooltip=hum_tooltip, key="hum_current")
        prec_current = input_with_skip("Current Precipitation (mm):", 0, 15000, 0, tooltip=prec_tooltip, key="prec_current")
        weather_data = {
            "temperature": temp_current,
            "humidity": hum_current,
            "wind_speed": wind_current,
            "precipitation": prec_current
        }

    remoteness = calculate_remoteness(latitude, longitude)
    if remoteness != -1:
        st.write(f"Remoteness (distance to nearest city): {remoteness:.2f} km")
    else:
        st.error("Could not calculate remoteness (distance to nearest city).")

    distance_to_station = calculate_distance_to_station(latitude, longitude)
    if distance_to_station is not None:
        st.write(f"Distance to Nearest Weather Station: {distance_to_station:.2f} km")
    else:
        st.error("Could not calculate distance to the nearest weather station.")

with tab3:
    # Historical Weather Data Options
    historical_weather_data = {}
    days_options = {"7 Days Ago": 7, "15 Days Ago": 15, "30 Days Ago": 30}
    selected_days = st.multiselect("Select Historical Weather Periods:", list(days_options.keys()))

    # Initialize historical variables with default values (in case data is skipped)
    temp_pre_7, wind_pre_7, hum_pre_7, prec_pre_7 = None, None, None, None
    temp_pre_15, wind_pre_15, hum_pre_15, prec_pre_15 = None, None, None, None
    temp_pre_30, wind_pre_30, hum_pre_30, prec_pre_30 = None, None, None, None

    # Fetch and process data only for selected days
    for day_label in selected_days:
        days_ago = days_options[day_label]
        target_date = (datetime.datetime.utcnow() - datetime.timedelta(days=days_ago)).strftime("%Y-%m-%d")
        historical_weather = fetch_historical_weather(latitude, longitude, target_date)
    
        # User choice for each historical period
        fetch_from_api = st.checkbox(f"Fetch {day_label} data from API", value=True, key=f"fetch_{day_label}_api")
        if fetch_from_api:
            historical_weather = fetch_historical_weather(latitude, longitude, target_date)
            if historical_weather:
                st.write(f"Weather {day_label}:")
                st.write(f"Temperature: {historical_weather['temperature']} °C")
                st.write(f"Humidity: {historical_weather['humidity']}%")
                st.write(f"Wind Speed: {historical_weather['wind_speed']} m/s")
                st.write(f"Precipitation: {historical_weather['precipitation']} mm")
                historical_weather_data[day_label] = historical_weather

                # Assign values dynamically based on user selection
                if day_label == "7 Days Ago":
                    temp_pre_7 = historical_weather["temperature"]
                    wind_pre_7 = historical_weather["wind_speed"]
                    hum_pre_7 = historical_weather["humidity"]
                    prec_pre_7 = historical_weather["precipitation"]
                
                if day_label == "15 Days Ago":
                    temp_pre_15 = historical_weather["temperature"]
                    wind_pre_15 = historical_weather["wind_speed"]
                    hum_pre_15 = historical_weather["humidity"]
                    prec_pre_15 = historical_weather["precipitation"]

                if day_label == "30 Days Ago":
                    temp_pre_30 = historical_weather["temperature"]
                    wind_pre_30 = historical_weather["wind_speed"]
                    hum_pre_30 = historical_weather["humidity"]
                    prec_pre_30 = historical_weather["precipitation"]

        else:
        
            temp= input_with_skip(f"Temperature {day_label} (°C):", -50, 50, 20,  key=f"temp_{day_label}")
            wind = input_with_skip(f"Wind Speed {day_label} (m/s):", 0, 25, 5, key=f"wind_{day_label}")
            hum = input_with_skip(f"Humidity {day_label} (%):", 0, 100, 50, key=f"hum_{day_label}")
            prec = input_with_skip(f"Precipitation {day_label} (mm):", 0, 15000, 0, key=f"prec_{day_label}")
        
            if day_label == "7 Days Ago":
                temp_pre_7 = temp
                wind_pre_7 = wind
                hum_pre_7 = hum
                prec_pre_7 = prec
            
            if day_label == "15 Days Ago":
                temp_pre_15 = temp
                wind_pre_15 = wind
                hum_pre_15 = hum
                prec_pre_15 = prec

            if day_label == "30 Days Ago":
                temp_pre_30 = temp
                wind_pre_30 = wind
                hum_pre_30 = hum
                prec_pre_30 = prec

with tab4:
    st.header("Make Prediction")
    input_form = st.form(key="prediction_form")
    # discovery_time = 14 # median
    # containment_time = 16 # median

    discovery_time = input_with_skip("Discovery Time (hour):", 0, 24, 14)
    containment_time = input_with_skip("Containment Time (hour):", 0, 24, 16)
    #fire_magnitude = st.slider("Fire Magnitude:", 0.1, 50.0, 1.0)  # Updated range


    # Calculate putout_time as the difference between containment_time and discovery_time
    putout_time = None
    if discovery_time is not None and containment_time is not None:
        putout_time = containment_time - discovery_time
        if putout_time < 0:
            putout_time += 24  # Adjust for cases where containment_time is on the next day


    # remoteness = input_with_skip("Remoteness:", 0.0, 1.0, 0.5)

    # Display wind categories as a reference


    #wind_category = input_with_skip("Containment Wind Speed Category:", options=["Calm ", "Breezy", "Windy", "Stormy"], default_value="Calm")
    temp_cont = input_with_skip("Containment Temperature (°C):", -50, 50, 25, tooltip=temp_tooltip, key="temp_cont")
    wind_cont =input_with_skip("Containment Wind Speed (m/s):", min_value=0, max_value=25, default_value=5, tooltip=wind_tooltip, key="wind_cont")
    hum_cont = input_with_skip("Containment Humidity (%):", 0, 100, 50, tooltip=hum_tooltip, key="hum_cont")
    prec_cont = input_with_skip("Containment Precipitation (mm):", 0, 15000, 50, tooltip=prec_tooltip, key="prec_cont")

    # Discovery Date
    discovery_date = st.date_input("Discovery Date", min_value=datetime.date(2000, 1, 1), max_value=datetime.date(2025, 12, 31), value=datetime.date(2015, 6, 15), key="discovery_date")

    # Extract year, month, and day for discovery date
    discovery_year = discovery_date.year
    discovery_month = discovery_date.month
    discovery_day = discovery_date.day

    skip_containment = st.checkbox("Skip", key="skip_containment")

    if not skip_containment:
        
        # User provides containment date via calendar input
        containment_date = st.date_input("Containment Date", min_value=discovery_date, value=datetime.date.today(), key="containment_date")
    # Extract year, month, and day for containment date
        containment_year = containment_date.year
        containment_month = containment_date.month
        containment_day = containment_date.day

        # Validate containment date
        if containment_date < discovery_date:
            st.error("Containment date must be after discovery date.")
    else:
        # Skip option is checked, default containment date is today
        containment_date = datetime.date.today()

        # Extract year, month, and day for containment date
        containment_year = containment_date.year
        containment_month = containment_date.month
        containment_day = containment_date.day

        # If containment date is earlier than discovery, adjust it
        if containment_date < discovery_date:
            containment_date = discovery_date
            containment_year = containment_date.year
            containment_month = containment_date.month
            containment_day = containment_date.day


    # Prepare DataFrame
    user_data = pd.DataFrame([[
        discovery_time, containment_time, latitude, longitude, putout_time,
        distance_to_station,  weather_data['temperature'] if 'weather_data' in locals() and weather_data else 20, 
        weather_data['wind_speed'] if 'weather_data' in locals() and weather_data else 5,
        weather_data['humidity'] if 'weather_data' in locals() and weather_data else 50,
        weather_data['precipitation'] if 'weather_data' in locals() and weather_data else 0,
        cause_code, state_code, vegetation_code, discovery_year, discovery_month, discovery_day,
        containment_year, containment_month, containment_day, temp_pre_7, wind_pre_7, hum_pre_7, prec_pre_7,
        temp_pre_15, wind_pre_15, hum_pre_15, prec_pre_15,
        temp_pre_30, wind_pre_30, hum_pre_30, prec_pre_30, 
        remoteness,  temp_cont, hum_cont, prec_cont, wind_cont
    ]], columns=[
        "DiscoveryTime", "ContainmentTime", "Latitude", "Longitude", "PutoutTime",
        "DistanceToStation", "Temperature", "WindSpeed", "Humidity", "Precipitation",
        "Cause","State", "Vegetation","DiscoveryYear", "DiscoveryMonth", "DiscoveryDay",
        "ContainmentYear", "ContainmentMonth", "ContainmentDay","Temp_pre_7", "Wind_pre_7", "Hum_pre_7", "Prec_pre_7", 
        "Temp_pre_15", "Wind_pre_15", "Hum_pre_15", "Prec_pre_15",
        "Temp_pre_30", "Wind_pre_30", "Hum_pre_30", "Prec_pre_30",
        "Remoteness", "Temp_cont", "Hum_cont", "Prec_cont", "Wind_cont"
    ])

    # Fill missing values with default values
    user_data.fillna({
        "DiscoveryTime": 14, "ContainmentTime": 16, 
        "Latitude": state_center[0], "Longitude": state_center[1], 
        "Temperature": 20, "WindSpeed": 5, "Humidity": 50, "Precipitation": 0,
        "Cause": 0, "State": state_code, "Vegetation": 0, "DiscoveryYear": 2020, "DiscoveryMonth": 6, "DiscoveryDay": 15,
        "ContainmentYear": 2020, "ContainmentMonth": 6, "ContainmentDay": 20, "Temp_pre_7": 20, "Wind_pre_7": 5, "Hum_pre_7": 50, "Prec_pre_7": 0,
        "Temp_pre_15": 20, "Wind_pre_15": 5, "Hum_pre_15": 50, "Prec_pre_15": 0, "Temp_pre_30": 20, "Wind_pre_30": 5, "Hum_pre_30": 50, "Prec_pre_30": 0,
        "Remoteness": 0.5, "Temp_cont": 25, "Hum_cont": 50, "Prec_cont": 10, "Wind_cont": 0
    }, inplace=True)

    if st.button("Predict Wildfire Size"):
        if selected_model == 'DNN Model':
            try:
                # Preprocess user input data
                user_data_preprocessed = pipeline_cls.transform(user_data)

                # Perform prediction
                predicted_class = predict(user_data)
                class_mapping = {0: "B", 1: "C", 2: "D", 3: "E", 4: "F", 5: "G"}
                predicted_class_label = class_mapping[predicted_class[0]]
                
                st.success(f"Predicted Wildfire Size Class: {predicted_class_label}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")

        elif selected_model == "MLP Model":
            try:
                # Process input data using the pipeline
                user_data_preprocessed = pipeline_cls.transform(user_data)

                # MLP model logic
                if user_data_preprocessed.shape[1] != 100:
                    st.error("Feature mismatch: The model expects 100 features.")
                    st.stop()

                # Predict using the MLP model
                prediction = model.predict(user_data_preprocessed)

                # Handle classification output
                if isinstance(prediction[0], str):
                    # Direct class label prediction
                    st.success(f"Predicted Wildfire Size Class: {prediction[0]}")
                elif isinstance(prediction[0], (int, np.integer)):
                    # Numeric prediction, map to class labels
                    class_labels = ['B', 'C', 'D', 'E', 'F', 'G']
                    if 0 <= int(prediction[0]) < len(class_labels):
                        prediction_label = class_labels[int(prediction[0])]
                        st.success(f"Predicted Wildfire Size Class: {prediction_label}")
                    else:
                        raise ValueError(f"Prediction index {prediction[0]} is out of bounds for class labels.")
                else:
                    raise ValueError(f"Unexpected prediction type: {type(prediction[0])}. Prediction: {prediction[0]}")

            except Exception as e:
                import traceback
                st.error(f"Error during prediction: {e}")
                st.error("Detailed stack trace:")
                st.error(traceback.format_exc())

        else:  # Handle GMM Model or any other
            try:
                # Process input data using the pipeline
                user_data_preprocessed = pipeline_cls.transform(user_data)

                # Extract DistanceToStation value from the user input
                distance_to_station = user_data['DistanceToStation'].iloc[0]

                # Dynamically calculate Slope_Class using the saved thresholds
                slope_class_pred = int(classify_slope(distance_to_station, thresholds))

                # Add Slope_Class as a new feature
                import numpy as np
                if isinstance(user_data_preprocessed, pd.DataFrame):
                    user_data_preprocessed = user_data_preprocessed.values  # Convert to numpy array if it is a DataFrame
                user_data_final = np.hstack((user_data_preprocessed, [[slope_class_pred]]))

                # Predict using the GMM model
                prediction = model.predict(user_data_final)
                

                # Convert regression magnitude to size class
                if isinstance(prediction[0], (float, np.float64)):
                    size_class = magnitude_to_size_class(prediction[0])  # Convert magnitude to size class
                    st.success(f"Predicted Fire Magnitude: {prediction[0]}")
                    st.success(f"Predicted Wildfire Size Class: {size_class}")
                elif isinstance(prediction[0], str):
                    st.success(f"Predicted Wildfire Size Class: {prediction[0]}")
                else:
                    raise ValueError(f"Unexpected prediction type: {type(prediction[0])}. Prediction: {prediction[0]}")

            except Exception as e:
                import traceback
                st.error(f"Error during prediction: {e}")
                st.error("Detailed stack trace:")
                st.error(traceback.format_exc())

