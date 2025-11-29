import streamlit as st
st.set_page_config(page_title="IoT ML Realtime Dashboard", layout="wide")

import pandas as pd
import numpy as np
import joblib
import json
import time
from datetime import datetime
import plotly.graph_objs as go
import paho.mqtt.client as mqtt

# ================================
# CONFIG
# ================================
MQTT_BROKER = st.secrets.get("MQTT_BROKER", "broker.emqx.io")
MQTT_PORT   = int(st.secrets.get("MQTT_PORT", 1883))
TOPIC_SENSOR = st.secrets.get("TOPIC_SENSOR", "iot/class/session5/sensor")
TOPIC_OUTPUT = st.secrets.get("TOPIC_OUTPUT", "iot/class/session5/output")
MODEL_PATH = st.secrets.get("MODEL_PATH", "iot_temp_model.pkl")

# ================================
# SESSION STATE
# ================================
if "logs" not in st.session_state:
    st.session_state.logs = []

if "last_data" not in st.session_state:
    st.session_state.last_data = None

if "mqtt_connected" not in st.session_state:
    st.session_state.mqtt_connected = False

# ================================
# LOAD MODEL
# ================================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# ================================
# MQTT CALLBACKS
# Thread-safe → NO UI TOUCH HERE
# ================================
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        st.session_state.mqtt_connected = True
        client.subscribe(TOPIC_SENSOR)
        print("MQTT Connected → subscribed")
    else:
        st.session_state.mqtt_connected = False
        print("MQTT failed:", rc)

def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        temp = float(data["temp"])
        hum = float(data["hum"])
    except:
        return

    # prediction
    pred = model.predict([[temp, hum]])[0]

    # append log
    row = {
        "ts": datetime.now().strftime("%H:%M:%S"),
        "temp": temp,
        "hum": hum,
        "pred": pred
    }
    st.session_state.logs.append(row)
    st.session_state.last_data = row

    # send auto alert
    if pred == "Panas":
        client.publish(TOPIC_OUTPUT, "ALERT_ON")
    else:
        client.publish(TOPIC_OUTPUT, "ALERT_OFF")

# ================================
# MQTT CLIENT (NO THREAD UI ISSUE)
# loop_start() is SAFE
# ================================
def init_mqtt():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()          # <<< PENTING → STABIL
    return client

if "mqtt_client" not in st.session_state:
    st.session_state.mqtt_client = init_mqtt()

# ================================
# UI DASHBOARD
# ================================
st.title("🔥 IoT ML Realtime Dashboard")

col1, col2 = st.columns(2)

with col1:
    st.subheader("MQTT Status")
    st.metric("Connected", "Yes" if st.session_state.mqtt_connected else "No")

    st.subheader("Last Data")
    if st.session_state.last_data:
        st.write(st.session_state.last_data)
    else:
        st.info("Waiting for realtime data...")

with col2:
    st.subheader("Realtime Chart")

    df = pd.DataFrame(st.session_state.logs)
    if not df.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["ts"], y=df["temp"], mode="lines+markers", name="Temp"))
        fig.add_trace(go.Scatter(x=df["ts"], y=df["hum"], mode="lines+markers", name="Humidity"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data yet.")

st.subheader("Log Data (Last 20)")
if not df.empty:
    st.dataframe(df.tail(20))
