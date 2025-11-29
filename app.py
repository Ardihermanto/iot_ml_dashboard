import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
import plotly.graph_objs as go
import paho.mqtt.client as mqtt

st.set_page_config(page_title="IoT ML Realtime Dashboard", layout="wide")

# -----------------------------------------------------------
# CONFIG
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
        st.error("❌ Error load model: " + str(e))
        return None

model = load_model(MODEL_PATH)

# -----------------------------------------------------------
# SESSION STATE
# -----------------------------------------------------------
for key, val in {
    "connected": False,
    "logs": [],
    "last": None,
    "override": None,
    "mqtt_client": None
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# -----------------------------------------------------------
# MQTT CALLBACKS
# -----------------------------------------------------------
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        st.session_state.connected = True
        client.subscribe(TOPIC_SENSOR)
    else:
        st.session_state.connected = False

def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        temp = float(data["temp"])
        hum = float(data["hum"])
    except:
        return  # ignore bad JSON

    ts = datetime.utcnow().isoformat()
    pred = "N/A"
    conf = None

    if model is not None:
        X = [[temp, hum]]
        pred = model.predict(X)[0]
        try:
            conf = float(np.max(model.predict_proba(X)))
        except:
            conf = None

    row = {"ts": ts, "temp": temp, "hum": hum, "pred": pred, "conf": conf}

    st.session_state.logs.append(row)
    st.session_state.last = row

    # Auto send back to ESP32
    if st.session_state.override is None:
        if pred == "Panas":
            client.publish(TOPIC_OUTPUT, "ALERT_ON")
        else:
            client.publish(TOPIC_OUTPUT, "ALERT_OFF")

# -----------------------------------------------------------
# START MQTT CLIENT (loop_start – safe)
# -----------------------------------------------------------
if st.session_state.mqtt_client is None:
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start()
        st.session_state.mqtt_client = client
    except:
        st.session_state.connected = False

# -----------------------------------------------------------
# UI
# -----------------------------------------------------------
st.title("IoT ML Realtime Dashboard")

left, right = st.columns([1, 2])

# LEFT SIDE — STATUS PANEL
with left:
    st.subheader("Connection Status")
    st.metric("MQTT Connected", "Yes" if st.session_state.connected else "No")
    st.metric("Broker", MQTT_BROKER)

    st.subheader("Last Reading")
    if st.session_state.last:
        st.write(pd.DataFrame([st.session_state.last]).T)
    else:
        st.info("Waiting for sensor data...")

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
        st.info("Auto control enabled again")

# RIGHT SIDE — CHART + TABLE
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
        st.info("No incoming data yet.")

    st.subheader("Recent Logs")
    if not df.empty:
        st.dataframe(df.iloc[::-1].head(20))

st.markdown("---")
st.write("Manual override:", st.session_state.override)
st.write("Total messages:", len(st.session_state.logs))
