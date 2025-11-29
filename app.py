###############################################################
# FINAL STREAMLIT SAFE VERSION (NO ASYNC, NO THREAD)
###############################################################

import streamlit as st
st.set_page_config(page_title="IoT ML Dashboard", layout="wide")

import paho.mqtt.client as mqtt
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
import plotly.graph_objs as go
import time

###############################################################
# CONFIG
###############################################################

MQTT_BROKER = st.secrets.get("MQTT_BROKER", "broker.emqx.io")
MQTT_PORT   = int(st.secrets.get("MQTT_PORT", 1883))
TOPIC_SENSOR = st.secrets.get("TOPIC_SENSOR", "iot/class/session5/sensor")
TOPIC_OUTPUT = st.secrets.get("TOPIC_OUTPUT", "iot/class/session5/output")
MODEL_PATH = st.secrets.get("MODEL_PATH", "iot_temp_model.pkl")

###############################################################
# LOAD MODEL
###############################################################

@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model()

###############################################################
# INIT SESSION STATE
###############################################################

if "logs" not in st.session_state:
    st.session_state.logs = []

if "last" not in st.session_state:
    st.session_state.last = None

if "mqtt_connected" not in st.session_state:
    st.session_state.mqtt_connected = False

###############################################################
# MQTT CLIENT SETUP (SYNC, SAFE)
###############################################################

mqtt_messages = []  # buffer aman

def on_connect(client, userdata, flags, rc):
    st.session_state.mqtt_connected = (rc == 0)
    if rc == 0:
        client.subscribe(TOPIC_SENSOR)

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        mqtt_messages.append(payload)
    except:
        pass

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

try:
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
except:
    st.session_state.mqtt_connected = False

###############################################################
# STREAMLIT UI
###############################################################

st.title("IoT ML Realtime Dashboard (Streamlit Cloud Safe Mode)")

left, right = st.columns([1, 2])

###############################################################
# PROCESS MQTT DATA ON EACH RERUN
###############################################################

client.loop(timeout=1.0)

while mqtt_messages:
    data = mqtt_messages.pop(0)
    temp = float(data["temp"])
    hum = float(data["hum"])
    ts = datetime.utcnow().isoformat()

    pred = "N/A"
    conf = None
    try:
        X = [[temp, hum]]
        pred = model.predict(X)[0]
        try:
            conf = float(np.max(model.predict_proba(X)))
        except:
            pass
    except:
        pred = "ERR"

    row = {"ts": ts, "temp": temp, "hum": hum, "pred": pred, "conf": conf}
    st.session_state.logs.append(row)
    st.session_state.last = row

    # publish to ESP32
    if pred == "Panas":
        client.publish(TOPIC_OUTPUT, "ALERT_ON")
    else:
        client.publish(TOPIC_OUTPUT, "ALERT_OFF")

###############################################################
# LEFT PANEL
###############################################################

with left:
    st.subheader("Connection Status")
    st.metric("MQTT Connected", "Yes" if st.session_state.mqtt_connected else "No")
    st.metric("Broker", MQTT_BROKER)

    st.subheader("Last Reading")
    if st.session_state.last:
        st.write(pd.DataFrame([st.session_state.last]).T)
    else:
        st.info("Waiting data...")

    st.subheader("Download Log")
    if st.button("Export CSV"):
        df = pd.DataFrame(st.session_state.logs)
        st.download_button("Download CSV",
                           df.to_csv(index=False),
                           "iot_log.csv")

###############################################################
# RIGHT PANEL
###############################################################

with right:
    st.subheader("Live Chart")

    df = pd.DataFrame(st.session_state.logs)

    if not df.empty:
        df_plot = df.tail(200)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_plot["ts"],
            y=df_plot["temp"],
            mode="lines+markers",
            name="Temperature"
        ))
        fig.add_trace(go.Scatter(
            x=df_plot["ts"],
            y=df_plot["hum"],
            mode="lines+markers",
            name="Humidity"
        ))

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No data")

    st.subheader("Recent Data")
    st.dataframe(df.tail(20))

###############################################################
# AUTO REFRESH
###############################################################

st.experimental_rerun()
