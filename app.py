# ================================================================
# IoT ML Realtime Dashboard (Stable Version for Streamlit Cloud)
# ================================================================

import streamlit as st
st.set_page_config(page_title="IoT ML Realtime Dashboard", layout="wide")

import time
import json
import joblib
import queue
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objs as go
import paho.mqtt.client as mqtt

# ================================================================
# CONFIGURATION
# ================================================================
MQTT_BROKER = st.secrets.get("MQTT_BROKER", "broker.hivemq.com")
MQTT_PORT   = int(st.secrets.get("MQTT_PORT", 1883))
TOPIC_SENSOR = st.secrets.get("TOPIC_SENSOR", "iot/class/session5/sensor")
TOPIC_OUTPUT = st.secrets.get("TOPIC_OUTPUT", "iot/class/session5/output")
MODEL_PATH = st.secrets.get("MODEL_PATH", "iot_temp_model.pkl")

# ================================================================
# SESSION STATE INITIALIZATION
# ================================================================
if "mqtt_connected" not in st.session_state:
    st.session_state.mqtt_connected = False

if "logs" not in st.session_state:
    st.session_state.logs = []

if "last" not in st.session_state:
    st.session_state.last = None

if "override" not in st.session_state:
    st.session_state.override = None

if "mqtt_started" not in st.session_state:
    st.session_state.mqtt_started = False

# ================================================================
# SAFE MQTT QUEUE (Thread-safe tempat data masuk)
# ================================================================
mqtt_queue = queue.Queue()

# ================================================================
# LOAD ML MODEL (SAFE CACHED)
# ================================================================
@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

model = load_model(MODEL_PATH)

# ================================================================
# MQTT CALLBACKS (HANYA MASUKKAN DATA → QUEUE, NO STREAMLIT)
# ================================================================
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        st.session_state.mqtt_connected = True
        client.subscribe(TOPIC_SENSOR)
    else:
        st.session_state.mqtt_connected = False

def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        mqtt_queue.put(data)
    except:
        pass

# ================================================================
# START MQTT (HANYA SEKALI)
# ================================================================
if not st.session_state.mqtt_started:
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        st.session_state.mqtt_client = client
        st.session_state.mqtt_started = True
    except Exception as e:
        st.error(f"❌ MQTT Error: {e}")

# ================================================================
# PROCESS QUEUE → ML PREDICTION (AMAN UNTUK STREAMLIT)
# ================================================================
def process_mqtt_queue():
    processed = False
    while not mqtt_queue.empty():
        msg = mqtt_queue.get()

        temp = float(msg.get("temp"))
        hum = float(msg.get("hum"))
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ML prediction
        pred = "N/A"
        conf = None
        if model is not None:
            X = [[temp, hum]]
            pred = model.predict(X)[0]
            try:
                conf = float(np.max(model.predict_proba(X)))
            except:
                conf = None

        row = {"ts": ts, "temp": temp, "hum": hum, "pred": pred, "conf": conf}
        st.session_state.logs.append(row)
        st.session_state.last = row

        # SEND OUTPUT BACK TO ESP32
        if st.session_state.override is None:
            if pred == "Panas":
                st.session_state.mqtt_client.publish(TOPIC_OUTPUT, "ALERT_ON")
            else:
                st.session_state.mqtt_client.publish(TOPIC_OUTPUT, "ALERT_OFF")

        processed = True

    return processed


# ================================================================
# MAIN UI
# ================================================================
st.title("🔥 IoT ML Realtime Dashboard (Stable Version)")

left, right = st.columns([1, 2])

# ---------------- LEFT PANEL ----------------
with left:
    st.subheader("MQTT Status")
    st.metric("Broker", MQTT_BROKER)
    st.metric("Connected", "Yes" if st.session_state.mqtt_connected else "No")

    st.subheader("Last Reading")
    if st.session_state.last:
        st.write(pd.DataFrame([st.session_state.last]).T)
    else:
        st.info("Waiting for sensor data...")

    st.subheader("Manual Override")
    col1, col2 = st.columns(2)
    if col1.button("Force ALERT ON"):
        st.session_state.override = "ON"
        st.session_state.mqtt_client.publish(TOPIC_OUTPUT, "ALERT_ON")

    if col2.button("Force ALERT OFF"):
        st.session_state.override = "OFF"
        st.session_state.mqtt_client.publish(TOPIC_OUTPUT, "ALERT_OFF")

    if st.button("Clear Override"):
        st.session_state.override = None

    if st.button("Save Log to CSV"):
        df = pd.DataFrame(st.session_state.logs)
        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="iot_realtime_logs.csv"
        )

# ---------------- RIGHT PANEL ----------------
with right:
    st.subheader("Realtime Chart")
    df_plot = pd.DataFrame(st.session_state.logs)

    if not df_plot.empty:
        df_plot = df_plot.tail(200)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_plot["ts"], y=df_plot["temp"],
                                 mode="lines+markers", name="Temperature"))
        fig.add_trace(go.Scatter(x=df_plot["ts"], y=df_plot["hum"],
                                 mode="lines+markers", name="Humidity"))

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data yet")

    st.subheader("Recent Logs")
    if st.session_state.logs:
        st.dataframe(pd.DataFrame(st.session_state.logs).iloc[::-1].head(20))


# ================================================================
# REALTIME LOOP (AMAN UNTUK STREAMLIT)
# ================================================================
client = st.session_state.mqtt_client
client.loop(timeout=1.0)

# process queue
process_mqtt_queue()

# rerun otomatis setiap 1 detik
time.sleep(1)
st.rerun()

