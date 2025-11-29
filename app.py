import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
import plotly.graph_objs as go
from paho.mqtt.client import Client as MQTTClient

st.set_page_config(page_title="IoT ML Realtime Dashboard", layout="wide")

# --------------------------
# CONFIG (WebSocket MQTT)
# --------------------------
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 8083
MQTT_PATH = "/mqtt"  # default ws path
USE_SSL = False

TOPIC_SENSOR = "iot/class/session5/sensor"
TOPIC_OUTPUT = "iot/class/session5/output"

MODEL_PATH = "iot_temp_model.pkl"

# --------------------------
# LOAD MODEL
# --------------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error("Model load error: " + str(e))
        return None

model = load_model()

# --------------------------
# SESSION STATE
# --------------------------
if "mqtt_connected" not in st.session_state:
    st.session_state.mqtt_connected = False
if "mqtt_client" not in st.session_state:
    st.session_state.mqtt_client = None
if "logs" not in st.session_state:
    st.session_state.logs = []
if "last" not in st.session_state:
    st.session_state.last = None
if "override" not in st.session_state:
    st.session_state.override = None

# --------------------------
# MQTT MESSAGE HANDLER
# --------------------------
def handle_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        temp = float(data["temp"])
        hum = float(data["hum"])
    except:
        return

    ts = datetime.utcnow().isoformat()
    pred = model.predict([[temp, hum]])[0]

    row = {
        "ts": ts,
        "temp": temp,
        "hum": hum,
        "pred": pred,
        "conf": None
    }

    st.session_state.logs.append(row)
    st.session_state.last = row

    # auto alert
    if st.session_state.override is None:
        if pred == "Panas":
            client.publish(TOPIC_OUTPUT, "ALERT_ON")
        else:
            client.publish(TOPIC_OUTPUT, "ALERT_OFF")

# --------------------------
# MQTT CONNECT (NO THREAD)
# --------------------------
def ensure_connected():
    if st.session_state.mqtt_client:
        return st.session_state.mqtt_client

    client = MQTTClient(transport="websockets")
    client.on_connect = lambda c, u, f, rc: st.session_state.update(mqtt_connected=(rc == 0))
    client.on_message = handle_message

    try:
        client.connect(MQTT_BROKER, MQTT_PORT, keepalive=30)
        client.subscribe(TOPIC_SENSOR)
        client.loop(timeout=0.1)   # NON-BLOCKING
        st.session_state.mqtt_client = client
    except:
        st.session_state.mqtt_connected = False

    return client

client = ensure_connected()

# --------------------------
# UI
# --------------------------
st.title("IoT ML Realtime Dashboard (WebSocket MQTT)")

col1, col2 = st.columns([1, 2])

# ========== LEFT PANEL ==========
with col1:
    st.subheader("MQTT Status")
    st.metric("Connected", "Yes" if st.session_state.mqtt_connected else "No")
    st.metric("Broker", MQTT_BROKER)
    st.metric("Port", MQTT_PORT)

    st.subheader("Last Data")
    if st.session_state.last:
        st.write(pd.DataFrame([st.session_state.last]).T)
    else:
        st.info("Waiting for data...")

    st.subheader("Manual Override")
    c1, c2 = st.columns(2)
    if c1.button("ALERT_ON"):
        st.session_state.override = "ON"
        client.publish(TOPIC_OUTPUT, "ALERT_ON")
    if c2.button("ALERT_OFF"):
        st.session_state.override = "OFF"
        client.publish(TOPIC_OUTPUT, "ALERT_OFF")
    if st.button("Clear Override"):
        st.session_state.override = None

# ========== RIGHT PANEL ==========
with col2:
    st.subheader("Live Chart")
    df = pd.DataFrame(st.session_state.logs)
    if not df.empty:
        df_recent = df.tail(200)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_recent["ts"], y=df_recent["temp"],
                                 mode="lines+markers", name="Temp"))
        fig.add_trace(go.Scatter(x=df_recent["ts"], y=df_recent["hum"],
                                 mode="lines+markers", name="Humidity"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data yet")

    st.subheader("Latest Logs")
    if not df.empty:
        st.dataframe(df.iloc[::-1].head(20))

st.markdown("---")
st.caption("Realtime IoT + ML Dashboard (WebSocket MQTT)")

# maintain MQTT polling
if client:
    client.loop(timeout=0.1)
