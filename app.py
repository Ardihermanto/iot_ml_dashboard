# ======================================================
# Streamlit IoT Dashboard — Fully Thread-Safe Version
# ======================================================

import streamlit as st
st.set_page_config(page_title="IoT ML Realtime Dashboard", layout="wide")

import pandas as pd
import numpy as np
import joblib
import json
import queue
import threading
import time
from datetime import datetime
import plotly.graph_objs as go
import paho.mqtt.client as mqtt

# ======================================================
# LOAD CONFIG (MAIN THREAD ONLY)
# ======================================================
CONFIG = {
    "MQTT_BROKER": st.secrets.get("MQTT_BROKER", "broker.hivemq.com"),
    "MQTT_PORT": int(st.secrets.get("MQTT_PORT", 1883)),
    "TOPIC_SENSOR": st.secrets.get("TOPIC_SENSOR", "iot/class/session5/sensor"),
    "TOPIC_OUTPUT": st.secrets.get("TOPIC_OUTPUT", "iot/class/session5/output"),
    "MODEL_PATH": st.secrets.get("MODEL_PATH", "iot_temp_model.pkl"),
    "CLIENT_ID": "streamlit_iot_client"
}

MQTT_BROKER = CONFIG["MQTT_BROKER"]
MQTT_PORT = CONFIG["MQTT_PORT"]
TOPIC_SENSOR = CONFIG["TOPIC_SENSOR"]
TOPIC_OUTPUT = CONFIG["TOPIC_OUTPUT"]
MODEL_PATH = CONFIG["MODEL_PATH"]
CLIENT_ID = CONFIG["CLIENT_ID"]

# ======================================================
# THREAD-SAFE QUEUE
# ======================================================
mqtt_queue = queue.Queue()

# ======================================================
# SESSION STATE
# ======================================================
if "logs" not in st.session_state:
    st.session_state.logs = []
if "last" not in st.session_state:
    st.session_state.last = None
if "override" not in st.session_state:
    st.session_state.override = None
if "mqtt_connected" not in st.session_state:
    st.session_state.mqtt_connected = False

# ======================================================
# LOAD ML MODEL
# ======================================================
@st.cache_resource
def load_model(path):
    return joblib.load(path)

model = load_model(MODEL_PATH)

# ======================================================
# MQTT CALLBACKS — NO STREAMLIT CALLS HERE
# ======================================================
def on_connect(client, userdata, flags, rc):
    # No Streamlit calls
    if rc == 0:
        client.subscribe(TOPIC_SENSOR)
        mqtt_queue.put({"system": "connected"})
    else:
        mqtt_queue.put({"system": "disconnected"})

def on_message(client, userdata, msg):
    # No Streamlit calls
    try:
        data = json.loads(msg.payload.decode())
        temp = float(data.get("temp"))
        hum = float(data.get("hum"))
    except:
        return

    mqtt_queue.put({
        "ts": datetime.utcnow().isoformat(),
        "temp": temp,
        "hum": hum
    })

# ======================================================
# MQTT BACKGROUND THREAD
# ======================================================
def mqtt_worker():
    while True:
        try:
            client = mqtt.Client(CLIENT_ID)
            client.on_connect = on_connect
            client.on_message = on_message

            client.connect(MQTT_BROKER, MQTT_PORT, 60)
            client.loop_forever()

        except Exception as e:
            mqtt_queue.put({"system": "disconnected"})
            time.sleep(2)

# START THREAD ONCE
if "mqtt_thread_started" not in st.session_state:
    threading.Thread(target=mqtt_worker, daemon=True).start()
    st.session_state.mqtt_thread_started = True

# ======================================================
# MAIN THREAD — PROCESS QUEUE SAFELY
# ======================================================
while not mqtt_queue.empty():
    item = mqtt_queue.get()

    # System messages (connect/disconnect)
    if "system" in item:
        st.session_state.mqtt_connected = (item["system"] == "connected")
        continue

    # Sensor messages
    ts = item["ts"]
    temp = item["temp"]
    hum = item["hum"]

    pred = model.predict([[temp, hum]])[0]

    row = {"ts": ts, "temp": temp, "hum": hum, "pred": pred}
    st.session_state.logs.append(row)
    st.session_state.last = row

# ======================================================
# UI
# ======================================================
st.title("IoT ML Realtime Dashboard")

left, right = st.columns([1,2])

with left:
    st.subheader("Status")
    st.metric("MQTT Broker", MQTT_BROKER)
    st.metric("MQTT Status", "Connected" if st.session_state.mqtt_connected else "Disconnected")

    st.subheader("Last Data")
    if st.session_state.last:
        st.write(pd.DataFrame([st.session_state.last]).T)

with right:
    st.subheader("Live Chart")
    df = pd.DataFrame(st.session_state.logs)
    if not df.empty:
        df = df.tail(200)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["ts"], y=df["temp"], name="Temp"))
        fig.add_trace(go.Scatter(x=df["ts"], y=df["hum"], name="Hum"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data yet")

st.write("Total messages:", len(st.session_state.logs))
