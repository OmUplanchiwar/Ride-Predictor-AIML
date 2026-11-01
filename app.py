import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# ======================
# PAGE CONFIG (ONLY ONCE)
# ======================
st.set_page_config(page_title="Ride Prediction", layout="wide")

# ======================
# CUSTOM CSS (FIXED)
# ======================
st.markdown("""
<style>

/* ANIMATED BACKGROUND */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(270deg, #020617, #0f172a, #020617);
    background-size: 600% 600%;
    animation: moveBG 12s ease infinite;
}

/* ANIMATION */
@keyframes moveBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* REMOVE DIM */
[data-testid="stAppViewContainer"]::before {
    background: none !important;
}

/* TITLE */
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #f97316;
    margin-bottom: 20px;
}

/* CARDS */
.card {
    background: rgba(15, 23, 42, 0.9);
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    border: 1px solid rgba(249,115,22,0.2);
    box-shadow: 0 0 20px rgba(249,115,22,0.15);
}

/* BUTTON - centered */
.stButton {
    display: flex;
    justify-content: center;
}
.stButton>button {
    background-color: #f97316;
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    border: none;
}

/* SLIDERS */
.stSlider div[data-baseweb="slider"] {
    color: #f97316;
}

</style>
""", unsafe_allow_html=True)

# ======================
# TITLE
# ======================
st.markdown('<div class="title">🚗 Ride Demand & Fare Predictor</div>', unsafe_allow_html=True)


# ======================
# LOAD DATA AND TRAIN MODELS
# ======================

@st.cache_data
def load_data():
    df = pd.read_csv("fairfare_ride_demand_dataset.csv")
    return df.ffill()

@st.cache_resource
def train_models(df):
    processed = pd.get_dummies(df, drop_first=True)

    X = processed.drop(["Final_Fare", "Demand_Score"], axis=1)
    y_fare = processed["Final_Fare"]
    y_demand = processed["Demand_Score"]

    fare_model = RandomForestRegressor(
        n_estimators=50,
        random_state=42,
        n_jobs=-1
    )
    demand_model = RandomForestRegressor(
        n_estimators=50,
        random_state=42,
        n_jobs=-1
    )

    fare_model.fit(X, y_fare)
    demand_model.fit(X, y_demand)

    return processed, X, fare_model, demand_model

df = load_data()
df, X, fare_model, demand_model = train_models(df)


# ======================
# STATS CARDS
# ======================
col1, col2, col3 = st.columns(3)

col1.markdown(f"""
<div class="card">
<h3>📊 Records</h3>
<h1 style="color:#ff7a18">{len(df)}</h1>
</div>
""", unsafe_allow_html=True)

col2.markdown(f"""
<div class="card">
<h3>🚗 Avg Fare</h3>
<h1 style="color:#ff7a18">{df['Final_Fare'].mean():.2f}</h1>
</div>
""", unsafe_allow_html=True)

col3.markdown(f"""
<div class="card">
<h3>📈 Avg Demand</h3>
<h1 style="color:#ff7a18">{df['Demand_Score'].mean():.2f}</h1>
</div>
""", unsafe_allow_html=True)

# ======================
# INPUT SECTION
# ======================
st.markdown("### Enter Ride Details")

col1, col2, col3 = st.columns(3)

distance = col1.slider("Distance (KM)", 1, 50, 5)
hour = col2.slider("Hour", 0, 23, 12)
traffic = col3.slider("Traffic Delay", 0, 60, 10)

# ======================
# PREDICTION
# ======================
input_data = X.iloc[0:1].copy()

if 'Ride_Distance_KM' in input_data.columns:
    input_data['Ride_Distance_KM'] = distance
if 'Hour_of_Day' in input_data.columns:
    input_data['Hour_of_Day'] = hour
if 'Traffic_Delay' in input_data.columns:
    input_data['Traffic_Delay'] = traffic

st.markdown("<br>", unsafe_allow_html=True)

# Centered button and results using columns
_, center_col, _ = st.columns([1, 2, 1])

with center_col:
    if st.button("🚀 Predict", use_container_width=True):
        fare = fare_model.predict(input_data)[0]
        demand = demand_model.predict(input_data)[0]

        st.markdown(f"""
        <div class="card" style="margin-top: 20px;">
            <h3 style="color:#ffffff;">💰 Predicted Fare</h3>
            <h1 style="color:#ff7a18;">₹{round(fare, 2)}</h1>
            <h3 style="color:#ffffff; margin-top:20px;">📊 Demand Score</h3>
            <h1 style="color:#ff7a18;">{round(demand, 2)}</h1>
        </div>
        """, unsafe_allow_html=True)



         # python -m streamlit run app.py