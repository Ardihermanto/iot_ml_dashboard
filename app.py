# ======================================================
# app.py — FINAL FIX (No Streamlit inside MQTT thread)
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
# LOAD CONFIG IN MAIN THREAD ONLY
# --------------------
# (Thread MQTT tidak boleh akses st.secrets!)
APP_CONFIG = {
    "MQTT_BROKER": st.secrets.get("MQTT_BROKER", "broker.hivemq.com"),
    "MQTT_PORT": int(st.secrets.get("MQTT_PORT", 1883)),
    "TOPIC_SENSOR": st.secrets.get("TOPIC_SENSOR", "iot/class/session5/sensor"),
    "TOPIC_OUTPUT": st.secrets.get("TOPIC_OUTPUT", "iot/class/session5/output"),
    "MODEL_PATH": st.secrets.get("MODEL_PATH", "iot_temp_model.pkl"),
    "CLIENT_ID": "streamlit_iot_client"
}

# Copy ke variabel biasa agar thread aman
MQTT_BROKER = APP_CONFIG["MQTT_BROKER"]
MQTT_PORT = APP_CONFIG["MQTT_PORT"]
TOPIC_SENSOR = APP_CONFIG["TOPIC_SENSOR"]
TOPIC_OUTPUT = APP_CONFIG["TOPIC_OUTPUT"]
MODEL_PATH = APP_CONFIG["MODEL_PATH"]
CLIENT_ID = APP_CONFIG["CLIENT_ID"]

# --------------------
# QUEUE (AMAN UNTUK THREAD)
# --------------------
mqtt_queue = queue.Queue()

# --------------------
# SESSION STATE
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
# LOAD MODEL (MAIN THREAD)
# --------------------
@st.cache_resource
def load_model(path):
    return joblib.load(path)

try:
    model = load_model(MODEL_PATH)
except:
    st.stop()

# --------------------
# MQTT CALLBACK (NO STREAMLIT!)
# --------------------
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        client.subscribe(TOPIC_SENSOR)
    # Tidak ada streamlit di sini

def on_message(client, userdata, msg):
    """Thread MQTT hanya kirim data ke QUEUE."""
    try:
        payload = msg.payload.decode()
        data = json.loads(payload)
        temp = float(data["temp"])
        hum = float(data["hum"])
    except:
        return

    mqtt_queue.put({
        "ts": datetime.utcnow().isoformat(),
        "temp": temp,
        "hum": hum
    })

# --------------------
# MQTT BACKGROUND THREAD (AMAN)
# --------------------
def mqtt_worker():
    while True:
        try:
            client = mqtt.Client(CLIENT_ID)
            client.on_connect = on_connect
            client.on_message = on_message

            client.connect(MQTT_BROKER, MQTT_PORT, 60)
            st.session_state.mqtt_client = client

            client.loop_forever()
        except:
            time.sleep(3)

if st.session_state.mqtt_client is None:
    threading.Thread(target=mqtt_worker, daemon=True).start()
    time.sleep(0.2)

# --------------------
# MAIN THREAD → PROCESS QUEUE (AMAN)
# --------------------
while not mqtt_queue.empty():
    msg = mqtt_queue.get()
    temp, hum, ts = msg["temp"], msg["hum"], msg["ts"]

    pred = model.predict([[temp, hum]])[0]

    row = {"ts": ts, "temp": temp, "hum": hum, "pred": pred}
    st.session_state.logs.append(row)
    st.session_state.last = row

    # kirim balik jika perlu
    if st.session_state.override is None:
        if st.session_state.mqtt_client:
            st.session_state.mqtt_client.publish(
                TOPIC_OUTPUT,
                "ALERT_ON" if pred == "Panas" else "ALERT_OFF"
            )

# --------------------
# UI SECTION
# --------------------
st.title("IoT ML Realtime Dashboard")

left, right = st.columns([1,2])

with left:
    st.subheader("Status")
    st.metric("MQTT Broker", MQTT_BROKER)
    st.metric("MQTT State", "Connected" if st.session_state.connected else "Disconnected")

    st.subheader("Last Reading")
    if st.session_state.last:
        st.write(pd.DataFrame([st.session_state.last]).T)
    else:
        st.info("Waiting ESP32 data...")

    st.subheader("Manual Override")
    c1, c2 = st.columns(2)
    if c1.button("Force ALERT_ON"):
        st.session_state.override = "ON"
        st.session_state.mqtt_client.publish(TOPIC_OUTPUT, "ALERT_ON")
    if c2.button("Force ALERT_OFF"):
        st.session_state.override = "OFF"
        st.session_state.mqtt_client.publish(TOPIC_OUTPUT, "ALERT_OFF")
    if st.button("Clear Override"):
        st.session_state.override = None

with right:
    st.subheader("Live Data")
    df = pd.DataFrame(st.session_state.logs)
    if not df.empty:
        df = df.tail(200)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["ts"], y=df["temp"], mode="lines+markers", name="Temp"))
        fig.add_trace(go.Scatter(x=df["ts"], y=df["hum"], mode="lines+markers", name="Hum"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data yet")

st.write("Total messages:", len(st.session_state.logs))
