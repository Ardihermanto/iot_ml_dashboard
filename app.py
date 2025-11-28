# app.py
import streamlit as st
st.set_page_config(page_title="IoT ML Realtime Dashboard", layout="wide")

import pandas as pd
import numpy as np
import joblib
import json
import threading
import time
from datetime import datetime
import plotly.graph_objs as go
import paho.mqtt.client as mqtt

# --------------------
# CONFIG (ubah sesuai kebutuhan)
# --------------------
MQTT_BROKER = st.secrets.get("MQTT_BROKER", "broker.hivemq.com")
MQTT_PORT   = int(st.secrets.get("MQTT_PORT", 1883))
TOPIC_SENSOR = st.secrets.get("TOPIC_SENSOR", "iot/class/session5/sensor")
TOPIC_OUTPUT = st.secrets.get("TOPIC_OUTPUT", "iot/class/session5/output")
MODEL_PATH = st.secrets.get("MODEL_PATH", "iot_temp_model.pkl")
CLIENT_ID = "streamlit_iot_client"

# --------------------
# SESSION STATE init
# --------------------
if 'logs' not in st.session_state:
    st.session_state.logs = []  # dict rows: {"ts","temp","hum","pred","conf"}
if 'connected' not in st.session_state:
    st.session_state.connected = False
if 'last' not in st.session_state:
    st.session_state.last = None
if 'mqtt_client' not in st.session_state:
    st.session_state.mqtt_client = None
if 'override' not in st.session_state:
    st.session_state.override = None  # "ON"/"OFF"/None

# --------------------
# Load model (safely)
# --------------------
@st.cache_resource
def load_model(path):
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model(MODEL_PATH)

# --------------------
# MQTT callbacks & worker
# --------------------
def on_connect(client, userdata, flags, rc):
    st.session_state.connected = (rc == 0)
    if rc == 0:
        client.subscribe(TOPIC_SENSOR)
        print("MQTT connected, subscribed to", TOPIC_SENSOR)
    else:
        print("MQTT connect failed rc=", rc)

def on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode()
        data = json.loads(payload)
        temp = float(data.get("temp"))
        hum = float(data.get("hum"))
    except Exception as e:
        print("Failed parse msg:", e, msg.payload)
        return

    ts = datetime.utcnow().isoformat()
    pred = "N/A"
    conf = None
    if model is not None:
        try:
            X = [[temp, hum]]
            pred = model.predict(X)[0]
            # try predict_proba if available
            try:
                conf = float(np.max(model.predict_proba(X)))
            except Exception:
                conf = None
        except Exception as e:
            print("Prediction error:", e)
            pred = "ERR"

    row = {"ts": ts, "temp": temp, "hum": hum, "pred": pred, "conf": conf}
    st.session_state.logs.append(row)
    st.session_state.last = row

    # Auto-send alert to ESP32 if critical AND no manual override
    if st.session_state.override is None:
        if pred == "Panas":
            client.publish(TOPIC_OUTPUT, "ALERT_ON")
        else:
            client.publish(TOPIC_OUTPUT, "ALERT_OFF")

def mqtt_thread():
    while True:
        try:
            client = mqtt.Client(CLIENT_ID)
            client.on_connect = on_connect
            client.on_message = on_message

            # optional auth from secrets
            user = st.secrets.get("MQTT_USER")
            pwd  = st.secrets.get("MQTT_PASS")
            if user and pwd:
                client.username_pw_set(user, pwd)

            client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
            st.session_state.mqtt_client = client
            client.loop_forever()
        except Exception as e:
            st.session_state.connected = False
            print("MQTT thread error:", e)
            time.sleep(5)  # reconnect backoff

# start background thread once
if st.session_state.mqtt_client is None:
    t = threading.Thread(target=mqtt_thread, daemon=True)
    t.start()
    time.sleep(0.1)

# --------------------
# UI
# --------------------
st.set_page_config(page_title="IoT ML Realtime Dashboard", layout="wide")
st.title("IoT ML Realtime Dashboard")

left, right = st.columns([1, 2])

with left:
    st.subheader("Status")
    conn_text = "Connected" if st.session_state.connected else "Disconnected"
    st.metric("MQTT Broker", MQTT_BROKER)
    st.metric("MQTT Status", conn_text)
    if st.session_state.last:
        st.markdown("**Last reading**")
        st.write(pd.DataFrame([st.session_state.last]).T)
    else:
        st.info("Waiting for sensor data...")

    st.subheader("Manual Override")
    col1, col2 = st.columns(2)
    if col1.button("Force ALERT ON"):
        st.session_state.override = "ON"
        # publish immediately
        if st.session_state.mqtt_client:
            st.session_state.mqtt_client.publish(TOPIC_OUTPUT, "ALERT_ON")
        st.success("Published ALERT_ON")
    if col2.button("Force ALERT OFF"):
        st.session_state.override = "OFF"
        if st.session_state.mqtt_client:
            st.session_state.mqtt_client.publish(TOPIC_OUTPUT, "ALERT_OFF")
        st.success("Published ALERT_OFF")
    if st.button("Clear override"):
        st.session_state.override = None
        st.info("Manual override cleared; auto alerts resumed")

    if st.button("Save Log to CSV"):
        if st.session_state.logs:
            df = pd.DataFrame(st.session_state.logs)
            fn = "iot_log_" + datetime.utcnow().strftime("%Y%m%d%H%M%S") + ".csv"
            df.to_csv(fn, index=False)
            st.download_button("Download CSV", df.to_csv(index=False).encode('utf-8'), file_name=fn)
        else:
            st.warning("No logs to save")

with right:
    st.subheader("Live Chart")
    df_plot = pd.DataFrame(st.session_state.logs)
    if not df_plot.empty:
        # keep the last N points
        df_plot = df_plot.tail(200)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_plot["ts"], y=df_plot["temp"], mode="lines+markers", name="Temperature (°C)"))
        fig.add_trace(go.Scatter(x=df_plot["ts"], y=df_plot["hum"], mode="lines+markers", name="Humidity (%)"))
        # color by prediction for temp markers
        if "pred" in df_plot.columns:
            colors = []
            for p in df_plot["pred"]:
                if p == "Panas":
                    colors.append("red")
                elif p == "Normal":
                    colors.append("green")
                else:
                    colors.append("blue")
            fig.update_traces(marker=dict(color=colors), selector=dict(mode="markers"))
        fig.update_layout(xaxis_title="timestamp", height=450)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No live data yet")

    st.subheader("Recent Data")
    if st.session_state.logs:
        st.dataframe(pd.DataFrame(st.session_state.logs).iloc[::-1].head(20))
    else:
        st.write("—")

# footer / small status
st.markdown("---")
st.write("Manual override:", st.session_state.override)
st.write("Total messages:", len(st.session_state.logs))

