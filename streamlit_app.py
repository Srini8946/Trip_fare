# streamlit_app.py

import streamlit as st
import numpy as np
import pickle
from math import radians, cos, sin, asin, sqrt
from datetime import datetime

# Load model
model = pickle.load(open('fare_predictor.pkl', 'rb'))

# Haversine distance function
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return 6371 * c

st.title("🚕 TripFare Predictor")
st.write("Estimate the fare for a New York City taxi ride")

# User input
pickup_lat = st.number_input("Pickup Latitude", value=40.761432)
pickup_lon = st.number_input("Pickup Longitude", value=-73.979815)
dropoff_lat = st.number_input("Dropoff Latitude", value=40.651311)
dropoff_lon = st.number_input("Dropoff Longitude", value=-73.880333)
passenger_count = st.slider("Passenger Count", 1, 6, 1)
pickup_date = st.date_input("Pickup Date", value=datetime.now().date())
pickup_time = st.time_input("Pickup Time", value=datetime.now().time())
datetime_input = datetime.combine(pickup_date, pickup_time)
payment_type = st.selectbox("Payment Type", ['Credit card', 'Cash'])

# Derived features
trip_distance = haversine(pickup_lon, pickup_lat, dropoff_lon, dropoff_lat)
pickup_hour = datetime_input.hour
pickup_day = datetime_input.weekday()
am_pm = 0 if pickup_hour >= 12 else 1  # 1 for am, 0 for pm
is_night = 1 if pickup_hour <= 5 or pickup_hour >= 22 else 0

# Create one-hot encodings
payment_type_cash = 1 if payment_type == 'Cash' else 0
payment_type_credit = 1 if payment_type == 'Credit card' else 0

# Prepare input
input_data = {
    'trip_distance': trip_distance,
    'passenger_count': passenger_count,
    'pickup_hour': pickup_hour,
    'pickup_day': pickup_day,
    'is_night': is_night,
    'am_pm_am': am_pm,
    'payment_type_Cash': 1 if payment_type == 'Cash' else 0
}

# Align with training features
vendor_id = 1  # or 0, depending on default logic or user input
vendor_id = st.selectbox("Vendor ID", [0, 1])  # Add this in user inputs section
# Create the input vector with 9 features
feature_vector = np.array([[ 
    trip_distance,
    passenger_count,
    pickup_hour,
    pickup_day,
    is_night,
    am_pm,
    payment_type_cash,
    payment_type_credit,
    vendor_id
]]).reshape(1, -1)

# Add both one-hot encoded payment types
payment_type_cash = 1 if payment_type == 'Cash' else 0
payment_type_credit = 1 if payment_type == 'Credit card' else 0

# Predict
if st.button("Predict Fare"):
    log_pred = model.predict(feature_vector)[0]
    pred_fare = np.expm1(log_pred)
    st.success(f"Estimated Fare: ${pred_fare:.2f}")
