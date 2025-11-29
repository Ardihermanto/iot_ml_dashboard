# app.py
import streamlit as st
st.set_page_config(page_title="IoT ML Realtime Dashboard", layout="wide")

import paho.mqtt.client as mqtt
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objs as go
import joblib

# -----------------------------
# CONFIG
# -----------------------------
MQTT_BROKER = st.secrets.get("MQTT_BROKER", "broker.emqx.io")
MQTT_PORT   = int(st.secrets.get("MQTT_PORT", 1883))
TOPIC_SENSOR = st.secrets.get("TOPIC_SENSOR", "iot/class/session5/sensor")
TOPIC_OUTPUT = st.secrets.get("TOPIC_OUTPUT", "iot/class/session5/output")
MODEL_PATH = st.secrets.get("MODEL_PATH", "iot_temp_model.pkl")

# -----------------------------
# SESSION STATE INIT
# -----------------------------
if "mqtt_client" not in st.session_state:
    st.session_state.mqtt_client = None
if "connected" not in st.session_state:
    st.session_state.connected = False
if "logs" not in st.session_state:
    st.session_state.logs = []
if "last" not in st.session_state:
    st.session_state.last = None

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

model = load_model(MODEL_PATH)

# -----------------------------
# MQTT CALLBACKS
# -----------------------------
def on_connect(client, userdata, flags, rc, properties=None):
    st.session_state.connected = (rc == 0)
    if rc == 0:
        client.subscribe(TOPIC_SENSOR)
    else:
        print("Failed connect:", rc)

def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        temp = float(data["temp"])
        hum  = float(data["hum"])
    except:
        print("Bad payload:", msg.payload)
        return

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    pred = model.predict([[temp, hum]])[0]
    try:
        conf = float(np.max(model.predict_proba([[temp, hum]])))
    except:
        conf = None

    row = {"ts": ts, "temp": temp, "hum": hum, "pred": pred, "conf": conf}
    st.session_state.logs.append(row)
    st.session_state.last = row

    # Auto alert
    if pred == "Panas":
        client.publish(TOPIC_OUTPUT, "ALERT_ON")
    else:
        client.publish(TOPIC_OUTPUT, "ALERT_OFF")

# -----------------------------
# START MQTT (ONLY ONCE)
# -----------------------------
if st.session_state.mqtt_client is None:
    client = mqtt.Client(protocol=mqtt.MQTTv5)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    st.session_state.mqtt_client = client

# -----------------------------
# MQTT POLLING (NO THREAD)
# -----------------------------
# Safe method: call loop() repeatedly in Streamlit reruns
st.session_state.mqtt_client.loop(timeout=0.1)

# -----------------------------
# UI LAYOUT
# -----------------------------
st.title("🌡 IoT ML Realtime Dashboard")

left, right = st.columns([1, 2])

# -----------------------------
# LEFT PANEL
# -----------------------------
with left:
    st.subheader("Connection Status")
    st.metric("MQTT Broker", MQTT_BROKER)
    st.metric("MQTT Connected", "Yes" if st.session_state.connected else "No")

    st.subheader("Last Reading")
    if st.session_state.last:
        st.write(pd.DataFrame([st.session_state.last]).T)
    else:
        st.info("Waiting for ESP32 data...")

    if st.button("Save Log to CSV"):
        df = pd.DataFrame(st.session_state.logs)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "iot_log.csv")
        st.success("CSV Ready!")

# -----------------------------
# RIGHT PANEL — LIVE CHART
# -----------------------------
with right:
    st.subheader("Live Chart")
    df = pd.DataFrame(st.session_state.logs)

    if not df.empty:
        df_tail = df.tail(200)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_tail["ts"], y=df_tail["temp"],
                                 mode="lines+markers", name="Temperature"))
        fig.add_trace(go.Scatter(x=df_tail["ts"], y=df_tail["hum"],
                                 mode="lines+markers", name="Humidity"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data received yet.")

st.markdown("---")
st.write("Total messages:", len(st.session_state.logs))
