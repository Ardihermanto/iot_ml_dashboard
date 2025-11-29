import streamlit as st
import json
import time
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import plotly.graph_objs as go
import paho.mqtt.client as mqtt

st.set_page_config(page_title="IoT ML Dashboard", layout="wide")

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
MQTT_BROKER = st.secrets.get("MQTT_BROKER", "broker.emqx.io")
MQTT_PORT   = int(st.secrets.get("MQTT_PORT", 1883))
TOPIC_SENSOR = st.secrets.get("TOPIC_SENSOR", "iot/class/session5/sensor")
TOPIC_OUTPUT = st.secrets.get("TOPIC_OUTPUT", "iot/class/session5/output")
MODEL_PATH = st.secrets.get("MODEL_PATH", "iot_temp_model.pkl")

# -------------------------------------------------------------------
# SESSION STATE INIT
# -------------------------------------------------------------------
if "logs" not in st.session_state:
    st.session_state.logs = []

if "last_data" not in st.session_state:
    st.session_state.last_data = None

if "mqtt_client" not in st.session_state:
    st.session_state.mqtt_client = None

if "connected" not in st.session_state:
    st.session_state.connected = False

# -------------------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# -------------------------------------------------------------------
# INIT MQTT CLIENT
# -------------------------------------------------------------------
def create_client():
    client = mqtt.Client(protocol=mqtt.MQTTv311)   # stabil untuk Streamlit Cloud

    def on_connect(c, userdata, flags, rc):
        if rc == 0:
            st.session_state.connected = True
            c.subscribe(TOPIC_SENSOR)
        else:
            st.session_state.connected = False

    def on_message(c, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            temp = float(payload["temp"])
            hum = float(payload["hum"])
        except:
            return

        X = [[temp, hum]]
        pred = model.predict(X)[0]

        row = {
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "temp": temp,
            "hum": hum,
            "pred": pred
        }

        st.session_state.last_data = row
        st.session_state.logs.append(row)

        # auto-response to ESP32
        if pred == "Panas":
            c.publish(TOPIC_OUTPUT, "ALERT_ON")
        else:
            c.publish(TOPIC_OUTPUT, "ALERT_OFF")

    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(MQTT_BROKER, MQTT_PORT, keepalive=30)
    client.loop_start()
    return client


# create client once
if st.session_state.mqtt_client is None:
    st.session_state.mqtt_client = create_client()


# -------------------------------------------------------------------
# UI
# -------------------------------------------------------------------
st.title("🔥 IoT ML Realtime Dashboard (Stable Version)")

left, right = st.columns([1, 2])

with left:
    st.subheader("MQTT Status")
    st.metric("Connected", "Yes" if st.session_state.connected else "No")
    st.metric("Broker", MQTT_BROKER)

    st.subheader("Last Data")
    if st.session_state.last_data:
        st.write(st.session_state.last_data)
    else:
        st.info("Waiting for data...")

    # Manual override
    st.subheader("Manual Override")
    if st.button("Force ALERT ON"):
        st.session_state.mqtt_client.publish(TOPIC_OUTPUT, "ALERT_ON")
        st.success("Sent ALERT_ON")

    if st.button("Force ALERT OFF"):
        st.session_state.mqtt_client.publish(TOPIC_OUTPUT, "ALERT_OFF")
        st.success("Sent ALERT_OFF")

    if st.button("Save CSV"):
        df = pd.DataFrame(st.session_state.logs)
        st.download_button("Download CSV", df.to_csv(index=False), "iot_log.csv")

with right:
    st.subheader("Live Chart")

    df = pd.DataFrame(st.session_state.logs)

    if not df.empty:
        df_tail = df.tail(200)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_tail["ts"], y=df_tail["temp"], mode="lines+markers", name="Temp"))
        fig.add_trace(go.Scatter(x=df_tail["ts"], y=df_tail["hum"], mode="lines+markers", name="Humidity"))

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Waiting for sensor data...")


# -------------------------------------------------------------------
# AUTO REFRESH EVERY 1 SECOND
# -------------------------------------------------------------------
time.sleep(1)
st.experimental_rerun()
