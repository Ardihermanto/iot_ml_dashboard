# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import queue
import time
from datetime import datetime
import paho.mqtt.client as mqtt
import plotly.graph_objs as go

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="IoT ML Dashboard (Queue MQTT)", layout="wide")
st.title("IoT ML Realtime Dashboard — Queue-based MQTT")

# -------------------------
# Config (use Streamlit secrets or defaults)
# -------------------------
MQTT_BROKER = st.secrets.get("MQTT_BROKER", "broker.hivemq.com")
MQTT_PORT   = int(st.secrets.get("MQTT_PORT", 1883))
TOPIC_SENSOR = st.secrets.get("TOPIC_SENSOR", "iot/class/session5/sensor")
TOPIC_OUTPUT = st.secrets.get("TOPIC_OUTPUT", "iot/class/session5/output")
MODEL_PATH = st.secrets.get("MODEL_PATH", "iot_temp_model.pkl")

# -------------------------
# Session state init
# -------------------------
if "logs" not in st.session_state:
    st.session_state.logs = []
if "last" not in st.session_state:
    st.session_state.last = None
if "mqtt_connected" not in st.session_state:
    st.session_state.mqtt_connected = False
if "override" not in st.session_state:
    st.session_state.override = None

# -------------------------
# Thread-safe queue shared between MQTT callback (thread) and main Streamlit (main thread)
# -------------------------
if "mqtt_queue" not in st.session_state:
    st.session_state.mqtt_queue = queue.Queue()

# -------------------------
# Load model safely (cached)
# -------------------------
@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model(MODEL_PATH)

# -------------------------
# MQTT callbacks (must NOT call Streamlit API here)
# only put parsed messages into st.session_state.mqtt_queue
# -------------------------
def _on_connect(client, userdata, flags, rc):
    # rc = 0 -> success
    # Do NOT call st.* here
    if rc == 0:
        client.subscribe(TOPIC_SENSOR)
        # put a simple status message to queue
        st.session_state.mqtt_queue.put({"_type": "status", "connected": True})
    else:
        st.session_state.mqtt_queue.put({"_type": "status", "connected": False, "rc": rc})

def _on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode()
        j = json.loads(payload)
        temp = float(j.get("temp"))
        hum = float(j.get("hum"))
    except Exception:
        # ignore bad payload
        return

    ts = datetime.utcnow().isoformat()
    # place raw sensor data into queue
    st.session_state.mqtt_queue.put({
        "_type": "sensor",
        "ts": ts,
        "temp": temp,
        "hum": hum
    })

# -------------------------
# Setup MQTT client once (will start loop_start thread)
# -------------------------
if "mqtt_client" not in st.session_state:
    client = mqtt.Client()
    client.on_connect = _on_connect
    client.on_message = _on_message
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        client.loop_start()  # starts background thread managed by paho
        st.session_state.mqtt_client = client
        # put initial status attempt
        st.session_state.mqtt_queue.put({"_type": "status", "connected": None})
    except Exception as e:
        st.error(f"Failed to connect to MQTT broker: {e}")
        st.session_state.mqtt_client = None
        st.session_state.mqtt_queue.put({"_type": "status", "connected": False, "error": str(e)})

# -------------------------
# Poll the queue and update session_state (main Streamlit thread)
# -------------------------
def drain_mqtt_queue():
    q = st.session_state.mqtt_queue
    updated = False
    while not q.empty():
        item = q.get()
        if not isinstance(item, dict):
            continue
        t = item.get("_type")
        if t == "status":
            connected = item.get("connected")
            if connected is not None:
                st.session_state.mqtt_connected = bool(connected)
            updated = True
        elif t == "sensor":
            ts = item["ts"]
            temp = item["temp"]
            hum = item["hum"]
            # do ML prediction here in main thread (safe)
            pred = "N/A"
            conf = None
            if model is not None:
                try:
                    X = [[temp, hum]]
                    pred = model.predict(X)[0]
                    try:
                        conf = float(np.max(model.predict_proba(X)))
                    except Exception:
                        conf = None
                except Exception:
                    pred = "ERR"
            row = {"ts": ts, "temp": temp, "hum": hum, "pred": pred, "conf": conf}
            st.session_state.logs.append(row)
            st.session_state.last = row
            # Auto-send output to ESP32 unless override active
            if st.session_state.override is None and st.session_state.mqtt_client:
                try:
                    if pred == "Panas":
                        st.session_state.mqtt_client.publish(TOPIC_OUTPUT, "ALERT_ON")
                    else:
                        st.session_state.mqtt_client.publish(TOPIC_OUTPUT, "ALERT_OFF")
                except Exception:
                    pass
            updated = True
    return updated

# call drain at top of app run (every rerun)
drain_mqtt_queue()

# -------------------------
# UI
# -------------------------
left, right = st.columns([1, 2])

with left:
    st.subheader("Connection")
    st.metric("Broker", MQTT_BROKER)
    st.metric("MQTT Connected", "Yes" if st.session_state.mqtt_connected else "No")

    st.subheader("Manual Override")
    c1, c2 = st.columns(2)
    if c1.button("Force ALERT ON"):
        st.session_state.override = "ON"
        if st.session_state.mqtt_client:
            st.session_state.mqtt_client.publish(TOPIC_OUTPUT, "ALERT_ON")
    if c2.button("Force ALERT OFF"):
        st.session_state.override = "OFF"
        if st.session_state.mqtt_client:
            st.session_state.mqtt_client.publish(TOPIC_OUTPUT, "ALERT_OFF")
    if st.button("Clear Override"):
        st.session_state.override = None

    st.subheader("Last Reading")
    if st.session_state.last:
        st.json(st.session_state.last)
    else:
        st.info("Waiting for data...")

    if st.button("Save log CSV"):
        if st.session_state.logs:
            df = pd.DataFrame(st.session_state.logs)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv, file_name="iot_log.csv")
        else:
            st.warning("No logs yet")

with right:
    st.subheader("Live chart")
    df = pd.DataFrame(st.session_state.logs)
    if not df.empty:
        df_plot = df.tail(200)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_plot["ts"], y=df_plot["temp"], mode="lines+markers", name="temp"))
        fig.add_trace(go.Scatter(x=df_plot["ts"], y=df_plot["hum"], mode="lines+markers", name="hum"))
        # color markers by pred if present
        if "pred" in df_plot.columns:
            colors = []
            for p in df_plot["pred"]:
                if p == "Panas": colors.append("red")
                elif p == "Normal": colors.append("green")
                else: colors.append("blue")
            fig.update_traces(marker=dict(color=colors), selector=dict(mode="markers"))
        fig.update_layout(height=450, xaxis_title="timestamp")
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Recent logs")
        st.dataframe(df.iloc[::-1].head(20))
    else:
        st.info("No data yet")

st.markdown("---")
st.write("Manual override:", st.session_state.override)
st.write("Total messages:", len(st.session_state.logs))

# keep UI responsive: optional small sleep (not required)
time.sleep(0.1)
