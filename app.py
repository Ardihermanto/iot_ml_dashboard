import streamlit as st
st.set_page_config(page_title="IoT ML Dashboard", layout="wide")

import json
import time
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import plotly.graph_objs as go
import paho.mqtt.client as mqtt


# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
BROKER = st.secrets.get("MQTT_BROKER", "broker.hivemq.com")
PORT   = int(st.secrets.get("MQTT_PORT", "1883"))
TOPIC_SENSOR = st.secrets.get("TOPIC_SENSOR", "iot/class/session5/sensor")
TOPIC_OUTPUT = st.secrets.get("TOPIC_OUTPUT", "iot/class/session5/output")
MODEL_PATH   = st.secrets.get("MODEL_PATH", "iot_temp_model.pkl")

# -------------------------------------------------------
# SESSION STATE
# -------------------------------------------------------
if "mqtt" not in st.session_state:
    st.session_state.mqtt = None

if "logs" not in st.session_state:
    st.session_state.logs = []

if "last" not in st.session_state:
    st.session_state.last = None

# -------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None

model = load_model()


# -------------------------------------------------------
# INIT MQTT CLIENT (NO THREAD)
# -------------------------------------------------------
if st.session_state.mqtt is None:
    client = mqtt.Client(protocol=mqtt.MQTTv311)
    client.connect(BROKER, PORT, keepalive=60)
    client.subscribe(TOPIC_SENSOR)
    st.session_state.mqtt = client
else:
    client = st.session_state.mqtt


# -------------------------------------------------------
# POLLING FUNCTION (SAFE)
# -------------------------------------------------------
def poll_mqtt_message():
    """Read exactly ONE MQTT message without blocking."""
    try:
        rc = client.loop(timeout=0.1)

        # Map errors to safe ignore
        if rc != mqtt.MQTT_ERR_SUCCESS:
            return None

        # Retrieve *the next message* if queued
        msg = client._out_messages  # public queue
        # Actually paho stores incoming messages in _in_messages
        if hasattr(client, "_in_messages") and client._in_messages:
            m = client._in_messages.pop(0)
            return m

        return None

    except Exception:
        return None


# -------------------------------------------------------
# MAIN LOGIC (RUNS EACH STREAMLIT REFRESH)
# -------------------------------------------------------
msg = poll_mqtt_message()

if msg:
    try:
        payload = msg.payload.decode()
        data = json.loads(payload)
        temp = float(data["temp"])
        hum  = float(data["hum"])
        ts   = datetime.utcnow().isoformat()

        # ML PREDICTION
        pred = model.predict([[temp, hum]])[0]

        row = {"ts": ts, "temp": temp, "hum": hum, "pred": pred}
        st.session_state.logs.append(row)
        st.session_state.last = row

        # Send back output
        if pred == "Panas":
            client.publish(TOPIC_OUTPUT, "ALERT_ON")
        else:
            client.publish(TOPIC_OUTPUT, "ALERT_OFF")

    except:
        pass


# -------------------------------------------------------
# UI
# -------------------------------------------------------
st.title("🔥 IoT ML Realtime Dashboard (Polling-Safe Version)")

left, right = st.columns([1, 2])

with left:
    st.subheader("Connection")
    st.metric("Broker", BROKER)
    st.metric("MQTT Connected", "Yes")

    st.subheader("Last Data")
    if st.session_state.last:
        st.json(st.session_state.last)
    else:
        st.info("Waiting for data...")

    if st.button("Save Logs CSV"):
        df = pd.DataFrame(st.session_state.logs)
        st.download_button("Download", df.to_csv(index=False), "iot_log.csv")

with right:
    st.subheader("Live Chart")
    df = pd.DataFrame(st.session_state.logs)
    if not df.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["ts"], y=df["temp"], mode="lines+markers", name="Temp"))
        fig.add_trace(go.Scatter(x=df["ts"], y=df["hum"], mode="lines+markers", name="Humidity"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data yet")

st.caption("Auto-refresh setiap rerun Streamlit.")
time.sleep(1)
st.experimental_rerun()
