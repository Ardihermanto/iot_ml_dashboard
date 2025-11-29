import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import time
from datetime import datetime
import plotly.graph_objs as go
import paho.mqtt.client as mqtt

st.set_page_config(page_title="IoT ML Dashboard", layout="wide")

# ---------------- CONFIG ----------------
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
TOPIC_SENSOR = "iot/class/session5/sensor"
TOPIC_OUTPUT = "iot/class/session5/output"

MODEL_PATH = "iot_temp_model.pkl"

# -------------- SESSION STATE ----------
if "mqtt" not in st.session_state:
    st.session_state.mqtt = None
if "connected" not in st.session_state:
    st.session_state.connected = False
if "logs" not in st.session_state:
    st.session_state.logs = []
if "last" not in st.session_state:
    st.session_state.last = None

# ---------------- LOAD MODEL ------------
@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Model gagal load: {e}")
        return None

model = load_model()

# -------------- MQTT FUNCTIONS ----------
def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        st.session_state.connected = True
        client.subscribe(TOPIC_SENSOR)
    else:
        st.session_state.connected = False

def on_message(client, userdata, msg):
    payload = msg.payload.decode()
    try:
        data = json.loads(payload)
        temp = float(data["temp"])
        hum = float(data["hum"])
    except:
        return

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Prediction
    pred = model.predict([[temp, hum]])[0]

    log = {"timestamp": ts, "temp": temp, "hum": hum, "prediction": pred}
    st.session_state.logs.append(log)
    st.session_state.last = log

    # Publish output
    if pred == "Panas":
        client.publish(TOPIC_OUTPUT, "ALERT_ON")
    else:
        client.publish(TOPIC_OUTPUT, "ALERT_OFF")


# -------------- INIT MQTT ---------------
if st.session_state.mqtt is None:
    client = mqtt.Client(protocol=mqtt.MQTTv5)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()
    st.session_state.mqtt = client


# ------------------- UI ------------------
st.title("IoT ML Realtime Dashboard")

left, right = st.columns([1, 2])

with left:
    st.subheader("Connection Status")
    st.metric("Connected", "Yes" if st.session_state.connected else "No")
    st.metric("Broker", MQTT_BROKER)

    st.subheader("Last Data")
    if st.session_state.last:
        st.json(st.session_state.last)
    else:
        st.info("Waiting for data...")

    if st.button("Save CSV"):
        df = pd.DataFrame(st.session_state.logs)
        st.download_button("Download", df.to_csv(index=False).encode("utf-8"), "logs.csv")


with right:
    st.subheader("Live Chart")

    df = pd.DataFrame(st.session_state.logs)
    if not df.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["temp"], name="Temperature"))
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["hum"], name="Humidity"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data yet")


# ---------------- AUTO REFRESH -----------
time.sleep(2)
st.experimental_rerun()
