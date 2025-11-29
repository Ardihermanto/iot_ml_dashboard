import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
import time
import plotly.graph_objs as go
import paho.mqtt.client as mqtt

# -----------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------
st.set_page_config(page_title="IoT ML Realtime Dashboard", layout="wide")

# -----------------------------------------------------------
# MQTT CONFIG
# -----------------------------------------------------------
MQTT_BROKER = st.secrets.get("MQTT_BROKER", "broker.emqx.io")
MQTT_PORT   = int(st.secrets.get("MQTT_PORT", 1883))
TOPIC_SENSOR = st.secrets.get("TOPIC_SENSOR", "iot/class/session5/sensor")
TOPIC_OUTPUT = st.secrets.get("TOPIC_OUTPUT", "iot/class/session5/output")
MODEL_PATH = st.secrets.get("MODEL_PATH", "iot_temp_model.pkl")

# -----------------------------------------------------------
# LOAD MODEL
# -----------------------------------------------------------
@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"❌ Model load error: {e}")
        return None

model = load_model(MODEL_PATH)

# -----------------------------------------------------------
# SESSION STATE
# -----------------------------------------------------------
if "connected" not in st.session_state:
    st.session_state.connected = False

if "logs" not in st.session_state:
    st.session_state.logs = []

if "last" not in st.session_state:
    st.session_state.last = None

if "override" not in st.session_state:
    st.session_state.override = None

# -----------------------------------------------------------
# MQTT CALLBACKS
# -----------------------------------------------------------
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT broker")
        st.session_state.connected = True
        client.subscribe(TOPIC_SENSOR)
    else:
        print("MQTT connect failed:", rc)
        st.session_state.connected = False

def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        temp = float(data["temp"])
        hum = float(data["hum"])
    except:
        print("Bad MQTT payload:", msg.payload)
        return

    ts = datetime.utcnow().isoformat()

    # Predict
    pred = "N/A"
    conf = None
    if model is not None:
        X = [[temp, hum]]
        pred = model.predict(X)[0]

        try:
            conf = float(np.max(model.predict_proba(X)))
        except:
            conf = None

    row = {
        "ts": ts,
        "temp": temp,
        "hum": hum,
        "pred": pred,
        "conf": conf
    }

    st.session_state.logs.append(row)
    st.session_state.last = row

    # Send to ESP32
    if st.session_state.override is None:
        if pred == "Panas":
            client.publish(TOPIC_OUTPUT, "ALERT_ON")
        else:
            client.publish(TOPIC_OUTPUT, "ALERT_OFF")

# -----------------------------------------------------------
# START MQTT CLIENT (NO THREAD, NO ASYNC)
# -----------------------------------------------------------
if "mqtt_client" not in st.session_state:
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()  # NON-BLOCKING — aman di Streamlit

    st.session_state.mqtt_client = client

# -----------------------------------------------------------
# UI
# -----------------------------------------------------------
st.title("IoT ML Realtime Dashboard 🚀")

left, right = st.columns([1, 2])

# LEFT PANEL
with left:
    st.subheader("Connection Status")
    st.metric("MQTT Connected", "Yes" if st.session_state.connected else "No")
    st.metric("Broker", MQTT_BROKER)

    if st.session_state.last:
        st.write(pd.DataFrame([st.session_state.last]).T)
    else:
        st.info("Waiting for data...")

    st.subheader("Manual Override")
    col1, col2 = st.columns(2)

    if col1.button("Force ALERT ON"):
        st.session_state.override = "ON"
        st.session_state.mqtt_client.publish(TOPIC_OUTPUT, "ALERT_ON")
        st.success("ALERT_ON sent")

    if col2.button("Force ALERT OFF"):
        st.session_state.override = "OFF"
        st.session_state.mqtt_client.publish(TOPIC_OUTPUT, "ALERT_OFF")
        st.success("ALERT_OFF sent")

    if st.button("Clear Override"):
        st.session_state.override = None
        st.info("Auto AI control reactivated")

# RIGHT PANEL
with right:
    st.subheader("Live Chart")

    df = pd.DataFrame(st.session_state.logs)

    if not df.empty:
        df_recent = df.tail(200)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_recent["ts"], y=df_recent["temp"], mode="lines+markers", name="Temperature"))
        fig.add_trace(go.Scatter(x=df_recent["ts"], y=df_recent["hum"], mode="lines+markers", name="Humidity"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data yet.")

    st.subheader("Recent Data")
    if not df.empty:
        st.dataframe(df.iloc[::-1].head(20))

# FOOTER
st.markdown("---")
st.write("Manual override:", st.session_state.override)
st.write("Total messages:", len(st.session_state.logs))
