# app.py (FINAL stable, no-thread, safe for Streamlit Cloud)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
import paho.mqtt.client as mqtt
import plotly.graph_objs as go
from collections import deque
from streamlit_autorefresh import st_autorefresh

# -------- Page config ----------
st.set_page_config(page_title="IoT ML Realtime Dashboard", layout="wide")
st.title("IoT ML Realtime Dashboard — Stable Final")

# -------- Config from secrets ----------
MQTT_BROKER = st.secrets.get("MQTT_BROKER", "broker.hivemq.com")
MQTT_PORT   = int(st.secrets.get("MQTT_PORT", 1883))
TOPIC_SENSOR = st.secrets.get("TOPIC_SENSOR", "iot/class/session5/sensor")
TOPIC_OUTPUT = st.secrets.get("TOPIC_OUTPUT", "iot/class/session5/output")
MODEL_PATH = st.secrets.get("MODEL_PATH", "iot_temp_model.pkl")

# -------- Session state init ----------
if "mqtt_client" not in st.session_state:
    st.session_state.mqtt_client = None
if "mqtt_connected" not in st.session_state:
    st.session_state.mqtt_connected = False
if "logs" not in st.session_state:
    st.session_state.logs = []       # list of dict rows
if "incoming" not in st.session_state:
    st.session_state.incoming = deque(maxlen=1000)  # messages pushed by callback
if "last" not in st.session_state:
    st.session_state.last = None
if "override" not in st.session_state:
    st.session_state.override = None

# -------- Load model (cached) ----------
@st.cache_resource
def load_model(path):
    try:
        m = joblib.load(path)
        return m
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model(MODEL_PATH)

# -------- MQTT callbacks (safe) ----------
def _on_connect(client, userdata, flags, rc):
    # This callback will be executed SYNCHRONOUSLY inside client.loop(...) calls.
    st.session_state.mqtt_connected = (rc == 0)
    if rc == 0:
        client.subscribe(TOPIC_SENSOR)

def _on_message(client, userdata, msg):
    # The payload is put into session_state.incoming (deque).
    # Callback runs inside the same thread when we call client.loop(timeout=...).
    try:
        payload = msg.payload.decode()
        data = json.loads(payload)
    except Exception:
        return
    st.session_state.incoming.append(data)

# -------- Create & connect client (once) ----------
if st.session_state.mqtt_client is None:
    client = mqtt.Client()
    client.on_connect = _on_connect
    client.on_message = _on_message
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        # DO NOT call loop_start() — we will call loop(...) manually below
        st.session_state.mqtt_client = client
    except Exception as e:
        st.error(f"Cannot connect to MQTT broker: {e}")
        st.session_state.mqtt_client = None
        st.session_state.mqtt_connected = False

# -------- Polling / autoreload ----------
# Use st_autorefresh to rerun the script every 1 second (1000 ms)
# so we call client.loop(timeout=...) each run and process incoming messages
st_autorefresh(interval=1000, key="autorefresh")  # rerun every 1s

# -------- On each rerun: process network events and incoming queue ----------
client = st.session_state.mqtt_client
if client is not None:
    try:
        # process network events (non-blocking, runs callbacks in this thread)
        client.loop(timeout=0.1)  # short single iteration
    except Exception:
        st.session_state.mqtt_connected = False

# Process items collected by callback (incoming deque)
while st.session_state.incoming:
    msg = st.session_state.incoming.popleft()
    # Expected format: {"temp": 29.5, "hum": 70.1}
    try:
        temp = float(msg.get("temp"))
        hum = float(msg.get("hum"))
    except Exception:
        continue

    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    # Predict (safe)
    pred = "N/A"
    conf = None
    if model is not None:
        try:
            pred = model.predict([[temp, hum]])[0]
            # optional confidence
            try:
                conf = float(np.max(model.predict_proba([[temp, hum]])))
            except Exception:
                conf = None
        except Exception:
            pred = "ERR"

    row = {"ts": ts, "temp": temp, "hum": hum, "pred": pred, "conf": conf}
    st.session_state.logs.append(row)
    st.session_state.last = row

    # Send output to ESP32 unless override set
    if client is not None and st.session_state.override is None:
        try:
            if pred == "Panas":
                client.publish(TOPIC_OUTPUT, "ALERT_ON")
            else:
                client.publish(TOPIC_OUTPUT, "ALERT_OFF")
        except Exception:
            pass

# -------- UI ----------
left, right = st.columns([1, 2])

with left:
    st.subheader("Connection")
    st.metric("MQTT Broker", MQTT_BROKER)
    st.metric("Connected", "Yes" if st.session_state.mqtt_connected else "No")
    st.metric("Messages", len(st.session_state.logs))

    st.subheader("Last Reading")
    if st.session_state.last:
        st.json(st.session_state.last)
    else:
        st.info("Waiting for data...")

    st.subheader("Manual Override")
    col1, col2 = st.columns(2)
    if col1.button("Force ALERT ON"):
        st.session_state.override = "ON"
        if client:
            client.publish(TOPIC_OUTPUT, "ALERT_ON")
            st.success("ALERT_ON published")
    if col2.button("Force ALERT OFF"):
        st.session_state.override = "OFF"
        if client:
            client.publish(TOPIC_OUTPUT, "ALERT_OFF")
            st.success("ALERT_OFF published")
    if st.button("Clear override"):
        st.session_state.override = None
        st.info("Auto alerts resumed")

    if st.button("Download logs CSV"):
        if st.session_state.logs:
            dfdl = pd.DataFrame(st.session_state.logs)
            csv = dfdl.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv, file_name="iot_logs.csv", mime="text/csv")
        else:
            st.warning("No logs yet")

with right:
    st.subheader("Live Chart (last 200)")
    if st.session_state.logs:
        df = pd.DataFrame(st.session_state.logs).tail(200)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["ts"], y=df["temp"], mode="lines+markers", name="Temperature (°C)"))
        fig.add_trace(go.Scatter(x=df["ts"], y=df["hum"], mode="lines+markers", name="Humidity (%)"))
        # color markers by prediction
        if "pred" in df.columns:
            colors = []
            for p in df["pred"]:
                if p == "Panas":
                    colors.append("red")
                elif p == "Normal":
                    colors.append("green")
                else:
                    colors.append("blue")
            fig.update_traces(marker=dict(color=colors), selector=dict(mode="markers"))
        fig.update_layout(xaxis_title="timestamp", height=480)
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Recent Logs")
        st.dataframe(df.iloc[::-1].head(20))
    else:
        st.info("No incoming data yet")

st.markdown("---")
st.caption("Notes: Streamlit polls MQTT once per rerun; st_autorefresh triggers reruns every 1s.")
