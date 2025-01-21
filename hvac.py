import streamlit as st
import pandas as pd
import numpy as np
import datetime
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pytz

# Constants remain the same as original
WEATHER_API_KEY = "YOUR_API_KEY"
ZONES = ["Zone 1", "Zone 2", "Zone 3"]
INDIAN_CITIES = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", 
                "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Lucknow"]

ZONE_MODES = {
    "COMFORT": {"temp_range": (20, 24), "energy_factor": 1.2},
    "ECO": {"temp_range": (18, 26), "energy_factor": 0.8},
    "AUTO": {"temp_range": (19, 25), "energy_factor": 1.0}
}
class Alert:
    def __init__(self):
        if 'alerts' not in st.session_state:
            st.session_state.alerts = []
        if 'energy_spikes' not in st.session_state:
            st.session_state.energy_spikes = [np.random.choice([True, False], p=[0.3, 0.7]) for _ in range(3)]

    def check_alerts(self, zones_data):
        current_alerts = []
        for i, usage in enumerate(zones_data['energy_usage']):
            if usage > 350 or st.session_state.energy_spikes[i]:
                current_alerts.append({
                    "zone": ZONES[i],
                    "message": f"⚠️ High energy usage in {ZONES[i]}: {usage:.1f} kWh",
                    "suggestions": [
                        "Switch to ECO mode",
                        "Optimize airflow settings",
                        f"Current settings causing {np.random.randint(15, 30)}% higher usage"
                    ]
                })
        st.session_state.alerts = current_alerts

    def display_alerts(self):
        if st.session_state.alerts:
            for alert in st.session_state.alerts:
                with st.expander(alert["message"]):
                    for suggestion in alert["suggestions"]:
                        st.write(f"- {suggestion}")
                    if st.button(f"Switch {alert['zone']} to ECO mode"):
                        st.success(f"Switched {alert['zone']} to ECO mode")
        else:
            st.success("All systems operating normally")
            
class WeatherService:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
    
    def get_weather(self, city):
        try:
            params = {
                'q': f"{city},IN",
                'appid': self.api_key,
                'units': 'metric'
            }
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            if response.status_code == 200:
                return {
                    'temperature': data['main']['temp'],
                    'humidity': data['main']['humidity'],
                    'condition': data['weather'][0]['main'],
                    'wind_speed': data['wind']['speed'],
                    'feels_like': data['main']['feels_like'],
                    'pressure': data['main']['pressure']
                }
            else:
                return self._get_fallback_weather()
        except:
            return self._get_fallback_weather()
    
    def _get_fallback_weather(self):
        return {
            'temperature': 25,
            'humidity': 65,
            'condition': 'Clear',
            'wind_speed': 3,
            'feels_like': 27,
            'pressure': 1013
        }

class HVACPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self._train_model()
    
    def _train_model(self):
        n_samples = 1000
        np.random.seed(42)
        
        X = np.random.rand(n_samples, 6)
        X[:, 0] = X[:, 0] * 20 + 10
        X[:, 1] = np.random.randint(0, 30, n_samples)
        X[:, 2] = np.random.randint(0, 24, n_samples)
        X[:, 3] = np.random.randint(0, 7, n_samples)
        X[:, 4] = np.random.uniform(30, 70, n_samples)
        X[:, 5] = np.random.uniform(0, 10, n_samples)
        
        y = 22 + (X[:, 0] - 20) * 0.3
        y += (X[:, 1] / 30) * 2
        y += np.sin(X[:, 2] * np.pi / 12) * 0.5
        y -= (X[:, 4] - 50) * 0.02
        y -= X[:, 5] * 0.1
        
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, y)
    
    def predict_optimal_temperature(self, weather_data, occupancy, time_of_day, day_of_week):
        features = np.array([[
            weather_data['temperature'],
            occupancy,
            time_of_day,
            day_of_week,
            weather_data['humidity'],
            weather_data['wind_speed']
        ]])
        features_scaled = self.scaler.transform(features)
        return max(18, min(26, self.model.predict(features_scaled)[0]))

def generate_realistic_energy_data():
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.datetime.now(ist)
    
    dates = pd.date_range(
        end=current_time,
        periods=24,
        freq='H'
    )
    
    base_usage = 250
    time_factors = np.array([
        0.7, 0.6, 0.5, 0.4, 0.4, 0.5,
        0.7, 1.0, 1.2, 1.1, 1.0,
        1.1, 1.2, 1.1, 1.0, 1.1,
        1.2, 1.3, 1.4, 1.3, 1.2,
        1.1, 0.9, 0.8
    ])
    
    usage = base_usage * time_factors
    usage += np.random.normal(0, 20, 24)
    
    return pd.DataFrame({
        'Time': dates,
        'Usage': usage
    })

def plot_energy_usage(data):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.lineplot(data=data, x='Time', y='Usage', color='#00ff00', linewidth=2)
    
    ax.set_facecolor('#1E1E2F')
    fig.patch.set_facecolor('#1E1E2F')
    
    ax.set_xlabel('Time', color='white', fontsize=12)
    ax.set_ylabel('Energy Usage (kWh)', color='white', fontsize=12)
    ax.tick_params(colors='white')
    
    plt.xticks(rotation=45)
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.title('24-Hour Energy Usage Overview', color='white', fontsize=14, pad=20)
    
    return fig


def custom_css():
    return """
    <style>
    .stApp {
        background-color: #1E1E2F;
        color: white;
    }
    .metric-card {
        background-color: #2C2C44;
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .weather-card {
        background: linear-gradient(135deg, #1E3A8A, #1E40AF);
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .control-card {
        background-color: #2C2C44;
        padding: 25px;
        border-radius: 15px;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .section-header {
        color: #00ff00;
        font-size: 24px;
        margin: 20px 0;
        font-weight: bold;
    }
    .status-good { color: #00ff00; }
    .status-warning { color: #ffff00; }
    .status-critical { color: #ff0000; }
    .about-card {
        background: linear-gradient(135deg, #2C2C44, #3C3C54);
        padding: 25px;
        border-radius: 15px;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .feature-card {
        background-color: #2C2C44;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #00ff00;
    }
    .team-member-card {
        background-color: #2C2C44;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
    }
    </style>
    """


def initialize_session_state():
    if "zones_data" not in st.session_state:
        st.session_state.zones_data = {
            "temperature": [22.0] * 3,
            "air_quality": [75.0] * 3,
            "energy_usage": [250.0] * 3,
            "occupancy": [10] * 3,
            "humidity": [50.0] * 3,
            "mode": ["AUTO"] * 3,
            "airflow": ["Auto"] * 3
        }

def display_zone_controls(zone_index, weather_data, predictor):
    zone = ZONES[zone_index]
    st.markdown(f"""<div class='control-card'><h2>{zone} Control Panel</h2></div>""", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
      
        st.subheader("Status")
        st.metric("Temperature", f"{st.session_state.zones_data['temperature'][zone_index]}°C")
        st.metric("Air Quality", f"{st.session_state.zones_data['air_quality'][zone_index]}")
        st.metric("Energy Usage", f"{st.session_state.zones_data['energy_usage'][zone_index]} kWh")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
     
        st.subheader("Controls")
        
        mode = st.selectbox(
            "Zone Mode",
            list(ZONE_MODES.keys()),
            key=f"mode_{zone_index}"
        )
        
        settings = ZONE_MODES[mode]
        if mode != "AUTO":
            # Fix: Convert temperature range values to float type
            min_temp = float(settings["temp_range"][0])
            max_temp = float(settings["temp_range"][1])
            current_temp = float(st.session_state.zones_data["temperature"][zone_index])
            
            temp = st.slider(
                "Temperature",
                min_value=min_temp,
                max_value=max_temp,
                value=current_temp,
                step=0.5,
                key=f"temp_{zone_index}"
            )
            
            airflow = st.select_slider(
                "Airflow",
                options=["Low", "Medium", "High", "Auto"],
                key=f"airflow_{zone_index}"
            )
            
            st.session_state.zones_data["temperature"][zone_index] = temp
            st.session_state.zones_data["airflow"][zone_index] = airflow
            st.session_state.zones_data["energy_usage"][zone_index] *= settings["energy_factor"]
        else:
            occupancy = st.session_state.zones_data["occupancy"][zone_index]
            time_of_day = datetime.datetime.now().hour
            day_of_week = datetime.datetime.now().weekday()
            
            optimal_temp = predictor.predict_optimal_temperature(
                weather_data, occupancy, time_of_day, day_of_week
            )
            
            st.info(f"AI Recommended Temperature: {optimal_temp:.1f}°C")
            
            if st.button(f"Apply AI Settings for {zone}"):
                st.session_state.zones_data["temperature"][zone_index] = optimal_temp
                st.success(f"Applied AI settings to {zone}")
        
       
def display_about_us():
    st.title("About Smart HVAC Control System")
    
    # Company Overview
    
    st.markdown("### 🏢 Company Overview")
    st.write("""
    Smart HVAC Control System is a cutting-edge solution developed by TechCool Solutions, 
    established in 2023. We specialize in creating intelligent building management systems 
    that optimize energy consumption while maintaining optimal comfort levels.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Key Features
    
    st.markdown("### ✨ Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
       
        st.markdown("#### 🤖 AI-Powered Optimization")
        st.write("Machine learning algorithms that adapt to usage patterns and weather conditions")
        
        
       
        st.markdown("#### 📊 Real-time Analytics")
        st.write("Comprehensive energy usage monitoring and performance metrics")
        
    
    with col2:
        
        st.markdown("#### 🌡️ Multi-zone Control")
        st.write("Independent temperature and airflow control for different zones")
        
        
        
        st.markdown("#### ⚡ Energy Efficiency")
        st.write("Smart modes and automated adjustments to minimize energy consumption")
       
    
    
    
    # Our Mission
    
    st.markdown("### 🎯 Our Mission")
    st.write("""
    Our mission is to revolutionize building climate control through innovative technology 
    that reduces energy consumption while maximizing comfort. We are committed to 
    contributing to a sustainable future by helping businesses and institutions optimize 
    their HVAC operations.
    """)
    
    
    
def main():
    st.set_page_config(page_title="Smart HVAC Control", layout="wide")
    st.markdown(custom_css(), unsafe_allow_html=True)
    initialize_session_state()
    
    weather_service = WeatherService(WEATHER_API_KEY)
    predictor = HVACPredictor()
    alert_system = Alert()
    
    st.sidebar.markdown("<h2 style='color: #00ff00;'>Smart HVAC Control</h2>", unsafe_allow_html=True)
    
    sidebar_section = st.sidebar.radio("Navigation", ["Location Settings", "Alerts", "About Us"])
    
    if sidebar_section == "Location Settings":
        city = st.sidebar.selectbox("Select City", INDIAN_CITIES)
        weather_data = weather_service.get_weather(city)
        
        st.sidebar.markdown("""
            <div class='weather-card'>
                <h3>Current Weather</h3>
                <p>Temperature: {temp}°C</p>
                <p>Humidity: {humidity}%</p>
                <p>Wind Speed: {wind} m/s</p>
                <p>Condition: {condition}</p>
            </div>
        """.format(
            temp=weather_data['temperature'],
            humidity=weather_data['humidity'],
            wind=weather_data['wind_speed'],
            condition=weather_data['condition']
        ), unsafe_allow_html=True)
        
        st.title("🌡️ Smart HVAC Control System")
        
        tabs = st.tabs(ZONES)
        for i, tab in enumerate(tabs):
            with tab:
                display_zone_controls(i, weather_data, predictor)
        
        st.markdown("<h2 class='section-header'>Energy Usage Overview</h2>", unsafe_allow_html=True)
        energy_data = generate_realistic_energy_data()
        fig = plot_energy_usage(energy_data)
        st.pyplot(fig)
        
    elif sidebar_section == "Alerts":
        st.title("System Alerts")
        alert_system.check_alerts(st.session_state.zones_data)
        alert_system.display_alerts()
    elif sidebar_section == "About Us":
        display_about_us()

if __name__ == "__main__":
    main()