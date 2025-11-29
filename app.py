import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime
import joblib
import plotly.graph_objs as go
import paho.mqtt.client as mqtt
import time

# -----------------------------
# STREAMLIT PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="IoT ML Dashboard", layout="wide")

st.title("🔥 IoT ML Realtime Dashboard")
st.write("Status monitoring suhu & kelembapan dengan Machine Learning")

# -----------------------------
# SECRETS (broker & model path)
# -----------------------------
MQTT_BROKER = st.secrets.get("MQTT_BROKER", "broker.hivemq.com")
MQTT_PORT   = int(st.secrets.get("MQTT_PORT", 1883))
TOPIC_SENSOR = st.secrets.get("TOPIC_SENSOR", "iot/class/session5/sensor")
TOPIC_OUTPUT = st.secrets.get("TOPIC_OUTPUT", "iot/class/session5/output")
MODEL_PATH = st.secrets.get("MODEL_PATH", "iot_temp_model.pkl")

# -----------------------------
# SESSION STATE
# -----------------------------
if "logs" not in st.session_state:
    st.session_state.logs = []
if "last_data" not in st.session_state:
    st.session_state.last_data = None
if "mqtt_connected" not in st.session_state:
    st.session_state.mqtt_connected = False

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Model failed to load: {e}")
        return None

model = load_model(MODEL_PATH)

# -----------------------------
# MQTT HANDLERS
# -----------------------------
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        st.session_state.mqtt_connected = True
        client.subscribe(TOPIC_SENSOR)
    else:
        st.session_state.mqtt_connected = False

def on_message(client, userdata, msg):
    payload = msg.payload.decode()
    try:
        data = json.loads(payload)
        temp = float(data.get("temp"))
        hum = float(data.get("hum"))
    except:
        return

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Prediction
    pred = "N/A"
    if model is not None:
        try:
            pred = model.predict([[temp, hum]])[0]
        except:
            pred = "ERR"

    row = {
        "ts": ts,
        "temp": temp,
        "hum": hum,
        "pred": pred
    }

    st.session_state.last_data = row
    st.session_state.logs.append(row)

    # Send output back to ESP32
    if pred == "Panas":
        client.publish(TOPIC_OUTPUT, "ALERT_ON")
    else:
        client.publish(TOPIC_OUTPUT, "ALERT_OFF")

# -----------------------------
# MQTT CLIENT (NO THREAD)
# -----------------------------
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

try:
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
except Exception as e:
    st.error(f"Gagal connect ke MQTT broker: {e}")

client.loop_start()

# -----------------------------
# UI — LEFT SIDE STATUS PANEL
# -----------------------------
left, right = st.columns([1, 2])

with left:
    st.subheader("MQTT Status")
    st.metric("Connected", "Yes" if st.session_state.mqtt_connected else "No")
    st.metric("Broker", MQTT_BROKER)
    st.metric("Port", MQTT_PORT)

    st.subheader("Last Data")
    if st.session_state.last_data:
        st.write(st.session_state.last_data)
    else:
        st.info("Waiting for data...")

# -----------------------------
# UI — RIGHT SIDE CHART
# -----------------------------
with right:
    st.subheader("Live Chart")

    df = pd.DataFrame(st.session_state.logs)

    if df.shape[0] > 0:
        df = df.tail(200)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["ts"], y=df["temp"],
            mode="lines+markers", name="Temperature"
        ))
        fig.add_trace(go.Scatter(
            x=df["ts"], y=df["hum"],
            mode="lines+markers", name="Humidity"
        ))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data yet")

    st.subheader("Latest Logs")
    if df.shape[0] > 0:
        st.dataframe(df.iloc[::-1].head(20))
    else:
        st.write("—")

