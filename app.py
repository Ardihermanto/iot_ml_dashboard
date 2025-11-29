import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import json
from datetime import datetime
import joblib
import plotly.graph_objs as go
import paho.mqtt.client as mqtt
import threading

# PAGE CONFIG
st.set_page_config(page_title="IoT ML Dashboard", layout="wide")

# AUTORERESH tiap 2 detik
st_autorefresh(interval=2000, key="data_refresh")

# ---------------------
# LOAD MODEL
# ---------------------
@st.cache_resource
def load_model():
    return joblib.load("iot_temp_model.pkl")

model = load_model()

# ---------------------
# SESSION STATE INIT
# ---------------------
if "logs" not in st.session_state:
    st.session_state.logs = []

if "mqtt_connected" not in st.session_state:
    st.session_state.mqtt_connected = False

if "mqtt_client" not in st.session_state:
    st.session_state.mqtt_client = None

# ---------------------
# MQTT CALLBACKS
# ---------------------
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        st.session_state.mqtt_connected = True
        client.subscribe(st.secrets["TOPIC_SENSOR"])
    else:
        st.session_state.mqtt_connected = False

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        temp = float(payload["temp"])
        hum  = float(payload["hum"])
        ts = datetime.utcnow().isoformat()

        pred = model.predict([[temp, hum]])[0]

        row = {
            "ts": ts,
            "temp": temp,
            "hum": hum,
            "pred": pred
        }

        st.session_state.logs.append(row)

        # Kirim kembali ke ESP32
        if pred == "Panas":
            client.publish(st.secrets["TOPIC_OUTPUT"], "ALERT_ON")
        else:
            client.publish(st.secrets["TOPIC_OUTPUT"], "ALERT_OFF")

    except Exception as e:
        print("Error message:", e)

# ---------------------
# MQTT BACKGROUND START
# ---------------------
def start_mqtt():
    if st.session_state.mqtt_client is None:
        client = mqtt.Client()
        client.on_connect = on_connect
        client.on_message = on_message
        client.connect(st.secrets["MQTT_BROKER"], int(st.secrets["MQTT_PORT"]), 60)
        st.session_state.mqtt_client = client
        client.loop_start()

threading.Thread(target=start_mqtt, daemon=True).start()

# ---------------------
# UI
# ---------------------
st.title("IoT Machine Learning Dashboard")

col1, col2 = st.columns([1,2])

with col1:
    st.subheader("Connection Status")

    st.metric("MQTT Connected",
              "Yes" if st.session_state.mqtt_connected else "No",
              delta=None)

    st.metric("Broker", st.secrets["MQTT_BROKER"])

    # Last Reading
    st.subheader("Last Data")
    if st.session_state.logs:
        st.write(pd.DataFrame([st.session_state.logs[-1]]).T)
    else:
        st.info("Waiting for data...")

with col2:
    st.subheader("Live Plot")

    if st.session_state.logs:
        df = pd.DataFrame(st.session_state.logs).tail(200)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["ts"], y=df["temp"], name="Temp"))
        fig.add_trace(go.Scatter(x=df["ts"], y=df["hum"], name="Humidity"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available yet.")

st.subheader("Recent Data Logs")
if st.session_state.logs:
    st.dataframe(pd.DataFrame(st.session_state.logs).iloc[::-1].head(20))
