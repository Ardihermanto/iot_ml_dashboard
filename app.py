# app.py
import streamlit as st
st.set_page_config(page_title="IoT ML Realtime Dashboard", layout="wide")

import json
import time
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import joblib
from datetime import datetime
import paho.mqtt.client as mqtt

# =================================================
# CONFIGURATION
# =================================================
MQTT_BROKER = st.secrets.get("MQTT_BROKER", "broker.emqx.io")
MQTT_PORT   = int(st.secrets.get("MQTT_PORT", 1883))
TOPIC_SENSOR = st.secrets.get("TOPIC_SENSOR", "iot/class/session5/sensor")
TOPIC_OUTPUT = st.secrets.get("TOPIC_OUTPUT", "iot/class/session5/output")
MODEL_PATH = st.secrets.get("MODEL_PATH", "iot_temp_model.pkl")

# =================================================
# SESSION STATE INIT
# =================================================
if "mqtt_client" not in st.session_state:
    st.session_state.mqtt_client = None
if "connected" not in st.session_state:
    st.session_state.connected = False
if "logs" not in st.session_state:
    st.session_state.logs = []
if "last" not in st.session_state:
    st.session_state.last = None

# =================================================
# MODEL LOADING
# =================================================
@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

model = load_model(MODEL_PATH)

# =================================================
# MQTT CALLBACKS (NEW v2 API)
# =================================================
def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        st.session_state.connected = True
        client.subscribe(TOPIC_SENSOR)
        print("MQTT Connected!")
    else:
        st.session_state.connected = False
        print("MQTT Connect failed:", reason_code)

def on_message(client, userdata, msg, properties):
    try:
        data = json.loads(msg.payload.decode())
        temp = float(data["temp"])
        hum  = float(data["hum"])
    except:
        print("Invalid payload:", msg.payload)
        return

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Prediction
    pred = model.predict([[temp, hum]])[0]
    try:
        conf = float(np.max(model.predict_proba([[temp, hum]])))
    except:
        conf = None

    row = {"ts": ts, "temp": temp, "hum": hum, "pred": pred, "conf": conf}

    st.session_state.logs.append(row)
    st.session_state.last = row

    # Auto alert logic
    if pred == "Panas":
        client.publish(TOPIC_OUTPUT, "ALERT_ON")
    else:
        client.publish(TOPIC_OUTPUT, "ALERT_OFF")

# =================================================
# MQTT START (NO THREAD, NO ASYNC)
# =================================================
if st.session_state.mqtt_client is None:
    client = mqtt.Client(protocol=mqtt.MQTTv5)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    st.session_state.mqtt_client = client

# Poll MQTT safely every Streamlit rerun
st.session_state.mqtt_client.loop(timeout=0.1)

# =================================================
# STREAMLIT UI
# =================================================
st.title("🌡 IoT ML Realtime Dashboard")

left, right = st.columns([1, 2])

# -------------------------------------------------
# LEFT PANEL
# -------------------------------------------------
with left:
    st.subheader("Connection Status")
    st.metric("MQTT Broker", MQTT_BROKER)
    st.metric("MQTT Connected", "Yes" if st.session_state.connected else "No")

    st.subheader("Last Reading")
    if st.session_state.last:
        st.write(pd.DataFrame([st.session_state.last]).T)
    else:
        st.info("Waiting for ESP32 data...")

    st.subheader("Save Log")
    if st.button("Download CSV"):
        df = pd.DataFrame(st.session_state.logs)
        st.download_button("Download", df.to_csv(index=False), "iot_log.csv")
        

# -------------------------------------------------
# RIGHT PANEL — LIVE CHART
# -------------------------------------------------
with right:
    st.subheader("Live Sensor Chart")

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
