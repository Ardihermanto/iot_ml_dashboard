import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
import plotly.graph_objs as go
import paho.mqtt.client as mqtt

st.set_page_config(page_title="IoT ML Dashboard", layout="wide")

# -------------------------
# CONFIG (edit sesuai kebutuhan)
# -------------------------
MQTT_BROKER = st.secrets.get("MQTT_BROKER", "broker.hivemq.com")
MQTT_PORT = int(st.secrets.get("MQTT_PORT", 1883))
TOPIC_SENSOR = st.secrets.get("TOPIC_SENSOR", "iot/class/session5/sensor")
TOPIC_OUTPUT = st.secrets.get("TOPIC_OUTPUT", "iot/class/session5/output")
MODEL_PATH = st.secrets.get("MODEL_PATH", "iot_temp_model.pkl")


# -------------------------
# Load Model
# -------------------------
@st.cache_resource
def load_model(path):
    return joblib.load(path)

try:
    model = load_model(MODEL_PATH)
    st.success("Model loaded successfully")
except:
    st.error("Failed to load model. Check MODEL_PATH")
    model = None


# -------------------------
# MQTT Simple Fetch Function
# -------------------------
def get_latest_message():
    """Fetch 1 message synchronously. No background thread."""

    client = mqtt.Client()
    client.connect(MQTT_BROKER, MQTT_PORT, 60)

    message_store = {"msg": None}

    def on_message(c, userdata, msg):
        message_store["msg"] = msg

    client.on_message = on_message
    client.subscribe(TOPIC_SENSOR)
    client.loop_start()

    # wait max 1 second
    timeout = datetime.now().timestamp() + 1
    while datetime.now().timestamp() < timeout:
        if message_store["msg"] is not None:
            break

    client.loop_stop()
    client.disconnect()

    return message_store["msg"]


# -------------------------
# Session State
# -------------------------
if "logs" not in st.session_state:
    st.session_state.logs = []

st.title("IoT ML Realtime Dashboard (Stable Mode)")


# -------------------------
# UI BUTTON – GET NEW DATA
# -------------------------
st.subheader("Fetch Latest Sensor Data")

if st.button("Get Data Now"):
    msg = get_latest_message()

    if msg is None:
        st.warning("No new message received from MQTT broker.")
    else:
        payload = msg.payload.decode()
        data = json.loads(payload)

        temp = float(data.get("temp"))
        hum = float(data.get("hum"))
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Predict
        if model:
            pred = model.predict([[temp, hum]])[0]
        else:
            pred = "N/A"

        # Save to session logs
        st.session_state.logs.append({
            "timestamp": ts,
            "temp": temp,
            "hum": hum,
            "prediction": pred
        })

        st.success(f"Received → Temp={temp}, Hum={hum}, Prediction={pred}")

        # publish output (simple)
        pub = mqtt.Client()
        pub.connect(MQTT_BROKER, MQTT_PORT, 60)
        if pred == "Panas":
            pub.publish(TOPIC_OUTPUT, "ALERT_ON")
        else:
            pub.publish(TOPIC_OUTPUT, "ALERT_OFF")
        pub.disconnect()


# -------------------------
# DISPLAY DATA
# -------------------------
st.subheader("Live Data Logs")

df = pd.DataFrame(st.session_state.logs)
st.dataframe(df.tail(20))

# -------------------------
# PLOT
# -------------------------
if not df.empty:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["temp"], mode="lines+markers", name="Temperature"))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["hum"], mode="lines+markers", name="Humidity"))
    st.plotly_chart(fig, use_container_width=True)


# -------------------------
# SAVE LOG
# -------------------------
if st.button("Save to CSV"):
    df.to_csv("log.csv", index=False)
    st.download_button("Download log.csv", df.to_csv(index=False).encode("utf-8"), "log.csv")
