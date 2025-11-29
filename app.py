# -------------------------------------------------------
# STREAMLIT MQTT + ML DASHBOARD (NO THREAD, NO ASYNCIO)
# -------------------------------------------------------
import streamlit as st
st.set_page_config(page_title="IoT ML Dashboard", layout="wide")

import time
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objs as go
from paho.mqtt import client as mqtt

# -------------------------------------------------------
# CONFIG (edit as needed)
# -------------------------------------------------------
BROKER = st.secrets.get("MQTT_BROKER", "broker.emqx.io")
PORT   = int(st.secrets.get("MQTT_PORT", 1883))
TOPIC_SENSOR = st.secrets.get("TOPIC_SENSOR", "iot/class/session5/sensor")
TOPIC_OUTPUT = st.secrets.get("TOPIC_OUTPUT", "iot/class/session5/output")
MODEL_PATH   = st.secrets.get("MODEL_PATH", "iot_temp_model.pkl")

# -------------------------------------------------------
# INIT SESSION STATE
# -------------------------------------------------------
if "logs" not in st.session_state:
    st.session_state.logs = []

if "last" not in st.session_state:
    st.session_state.last = None

if "mqtt" not in st.session_state:
    st.session_state.mqtt = None

if "connected" not in st.session_state:
    st.session_state.connected = False

# -------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------
@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None

model = load_model(MODEL_PATH)

# -------------------------------------------------------
# MQTT CONNECT (ONCE)
# -------------------------------------------------------
if st.session_state.mqtt is None:
    def on_connect(client, userdata, flags, rc):
        st.session_state.connected = (rc == 0)
        if rc == 0:
            client.subscribe(TOPIC_SENSOR)

    st.session_state.mqtt = mqtt.Client(protocol=mqtt.MQTTv5)
    st.session_state.mqtt.on_connect = on_connect
    st.session_state.mqtt.connect(BROKER, PORT, keepalive=30)
    st.session_state.mqtt.loop_start()
    time.sleep(1)

# -------------------------------------------------------
# READ ONE MESSAGE (NON-BLOCKING)
# -------------------------------------------------------
def poll_message():
    msg = st.session_state.mqtt._sock_recv()
    if not msg:
        return None

    # Paho MQTT v5 internal parser:
    try:
        packet = st.session_state.mqtt._packet_read(msg)
        if packet is None or not hasattr(packet, "topic"):
            return None

        payload = packet.payload.decode()
        return {
            "topic": packet.topic,
            "payload": payload
        }
    except:
        return None

# -------------------------------------------------------
# UI TITLE
# -------------------------------------------------------
st.title("IoT ML Realtime Dashboard (Stable)")

col1, col2 = st.columns([1,2])

with col1:
    st.subheader("Status")
    st.metric("Connected", "Yes" if st.session_state.connected else "No")
    st.metric("Broker", BROKER)

    if st.session_state.last:
        st.subheader("Last Data")
        st.write(pd.DataFrame([st.session_state.last]).T)
    else:
        st.info("Waiting for data...")

    # Save log
    if st.button("Save Log"):
        df = pd.DataFrame(st.session_state.logs)
        st.download_button("Download CSV", df.to_csv(index=False), "log.csv")

with col2:
    st.subheader("Live Chart")
    df = pd.DataFrame(st.session_state.logs)
    if not df.empty:
        df_plot = df.tail(200)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_plot["ts"], y=df_plot["temp"], mode="lines+markers", name="Temp"))
        fig.add_trace(go.Scatter(x=df_plot["ts"], y=df_plot["hum"], mode="lines+markers", name="Hum"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No chart data yet")

# -------------------------------------------------------
# REALTIME LOOP SECTION
# -------------------------------------------------------
st.subheader("Realtime Listener")

placeholder = st.empty()

# Run loop safely in Streamlit (auto-rerun)
for _ in range(15):  # 15 polls (≈15 seconds)
    msg = poll_message()

    if msg and TOPIC_SENSOR in msg["topic"]:
        try:
            data = json.loads(msg["payload"])
            temp = float(data["temp"])
            hum  = float(data["hum"])
            ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # ML inference
            pred = model.predict([[temp, hum]])[0]

            row = {"ts": ts, "temp": temp, "hum": hum, "pred": pred}
            st.session_state.logs.append(row)
            st.session_state.last = row

            # publish output
            out = "ALERT_ON" if pred == "Panas" else "ALERT_OFF"
            st.session_state.mqtt.publish(TOPIC_OUTPUT, out)

            placeholder.write(f"New data: {row}")

        except:
            pass

    time.sleep(1)

st.experimental_rerun()
