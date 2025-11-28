# ======================================================
# app.py — FIXED VERSION (AMAN UNTUK STREAMLIT CLOUD)
# ======================================================

import streamlit as st
st.set_page_config(page_title="IoT ML Realtime Dashboard", layout="wide")

import pandas as pd
import numpy as np
import joblib
import json
import threading
import time
from datetime import datetime
import queue
import plotly.graph_objs as go
import paho.mqtt.client as mqtt

# --------------------
# CONFIG
# --------------------
MQTT_BROKER = st.secrets.get("MQTT_BROKER", "broker.hivemq.com")
MQTT_PORT   = int(st.secrets.get("MQTT_PORT", 1883))
TOPIC_SENSOR = st.secrets.get("TOPIC_SENSOR", "iot/class/session5/sensor")
TOPIC_OUTPUT = st.secrets.get("TOPIC_OUTPUT", "iot/class/session5/output")
MODEL_PATH = st.secrets.get("MODEL_PATH", "iot_temp_model.pkl")
CLIENT_ID = "streamlit_iot_client"

# --------------------
# QUEUE untuk thread MQTT
# --------------------
mqtt_queue = queue.Queue()

# --------------------
# SESSION STATE init
# --------------------
if "logs" not in st.session_state:
    st.session_state.logs = []
if "connected" not in st.session_state:
    st.session_state.connected = False
if "last" not in st.session_state:
    st.session_state.last = None
if "mqtt_client" not in st.session_state:
    st.session_state.mqtt_client = None
if "override" not in st.session_state:
    st.session_state.override = None

# --------------------
# Load ML model
# --------------------
@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model(MODEL_PATH)

# --------------------
# MQTT CALLBACKS
# --------------------
def on_connect(client, userdata, flags, rc):
    st.session_state.connected = (rc == 0)
    if rc == 0:
        client.subscribe(TOPIC_SENSOR)
        print("MQTT Connected, subscribed:", TOPIC_SENSOR)
    else:
        print("MQTT connect failed rc=", rc)

def on_message(client, userdata, msg):
    """Callback thread — hanya masukkan ke QUEUE"""
    try:
        payload = msg.payload.decode()
        data = json.loads(payload)
        temp = float(data.get("temp"))
        hum = float(data.get("hum"))
        ts = datetime.utcnow().isoformat()

        mqtt_queue.put({
            "ts": ts,
            "temp": temp,
            "hum": hum
        })
    except Exception as e:
        print("Parse failed:", e)

# --------------------
# MQTT THREAD (background)
# --------------------
def mqtt_thread():
    while True:
        try:
            client = mqtt.Client(CLIENT_ID)
            client.on_connect = on_connect
            client.on_message = on_message

            client.connect(MQTT_BROKER, MQTT_PORT, 60)
            st.session_state.mqtt_client = client
            client.loop_forever()
        except Exception as e:
            st.session_state.connected = False
            print("MQTT Thread Error:", e)
            time.sleep(3)

# Start thread once
if st.session_state.mqtt_client is None:
    threading.Thread(target=mqtt_thread, daemon=True).start()
    time.sleep(0.2)

# --------------------
# UI
# --------------------
st.title("IoT ML Realtime Dashboard")

left, right = st.columns([1, 2])

# --------------------
# PROCESS QUEUE (MAIN THREAD)
# --------------------
while not mqtt_queue.empty():
    msg = mqtt_queue.get()
    temp = msg["temp"]
    hum  = msg["hum"]
    ts   = msg["ts"]

    # ML prediction in main thread
    pred = model.predict([[temp, hum]])[0]

    # update session logs
    row = {
        "ts": ts,
        "temp": temp,
        "hum": hum,
        "pred": pred
    }

    st.session_state.logs.append(row)
    st.session_state.last = row

    # Auto-send alert jika tidak override
    if st.session_state.override is None:
        if pred == "Panas":
            st.session_state.mqtt_client.publish(TOPIC_OUTPUT, "ALERT_ON")
        else:
            st.session_state.mqtt_client.publish(TOPIC_OUTPUT, "ALERT_OFF")

# --------------------
# LEFT PANEL
# --------------------
with left:
    st.subheader("Status")
    st.metric("MQTT Broker", MQTT_BROKER)
    st.metric("MQTT Status", "Connected" if st.session_state.connected else "Disconnected")

    if st.session_state.last:
        st.write("**Last Reading**")
        st.write(pd.DataFrame([st.session_state.last]).T)
    else:
        st.info("Waiting for sensor data...")

    st.subheader("Manual Override")
    c1, c2 = st.columns(2)
    if c1.button("Force ALERT_ON"):
        st.session_state.override = "ON"
        st.session_state.mqtt_client.publish(TOPIC_OUTPUT, "ALERT_ON")
        st.success("Sent ALERT_ON")

    if c2.button("Force ALERT_OFF"):
        st.session_state.override = "OFF"
        st.session_state.mqtt_client.publish(TOPIC_OUTPUT, "ALERT_OFF")
        st.success("Sent ALERT_OFF")

    if st.button("Clear Override"):
        st.session_state.override = None
        st.info("Auto alert resumed")

# --------------------
# RIGHT PANEL — CHART & TABLE
# --------------------
with right:
    st.subheader("Live Chart")
    df_plot = pd.DataFrame(st.session_state.logs)

    if not df_plot.empty:
        df_plot = df_plot.tail(200)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_plot["ts"], y=df_plot["temp"], mode="lines+markers", name="Temperature"))
        fig.add_trace(go.Scatter(x=df_plot["ts"], y=df_plot["hum"], mode="lines+markers", name="Humidity"))

        # color code
        colors = ["red" if p=="Panas" else "green" if p=="Normal" else "blue"
                  for p in df_plot["pred"]]
        fig.update_traces(marker=dict(color=colors), selector=dict(mode="markers"))

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data yet.")

    st.subheader("Recent Data")
    if st.session_state.logs:
        st.dataframe(df_plot.iloc[::-1].head(20))

st.markdown("---")
st.write("Manual override:", st.session_state.override)
st.write("Total messages:", len(st.session_state.logs))
