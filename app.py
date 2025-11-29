import streamlit as st
st.set_page_config(page_title="IoT ML Dashboard", layout="wide")

import paho.mqtt.client as mqtt
import pandas as pd
import numpy as np
import joblib
import json
import time
from datetime import datetime

# ----------------------
# CONFIG
# ----------------------
BROKER = st.secrets.get("MQTT_BROKER", "broker.hivemq.com")
PORT   = int(st.secrets.get("MQTT_PORT", 1883))
TOPIC_SENSOR = st.secrets.get("TOPIC_SENSOR", "iot/class/session5/sensor")
TOPIC_OUTPUT = st.secrets.get("TOPIC_OUTPUT", "iot/class/session5/output")
MODEL_PATH = st.secrets.get("MODEL_PATH", "iot_temp_model.pkl")

# ----------------------
# SESSION INIT
# ----------------------
if "client" not in st.session_state:
    st.session_state.client = mqtt.Client()
    st.session_state.client.connect(BROKER, PORT, 60)
    st.session_state.client.subscribe(TOPIC_SENSOR)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "model" not in st.session_state:
    st.session_state.model = joblib.load(MODEL_PATH)

model = st.session_state.model
client = st.session_state.client

# ----------------------
# POLLING MQTT
# ----------------------
def poll_mqtt_message():
    client.loop(timeout=0.1)  # process network events once

    rc, msg = client._easy_logical_read()
    if msg is None:
        return None

    try:
        data = json.loads(msg.payload.decode())
        return data
    except:
        return None


# ----------------------
# MAIN LOOP (every rerun)
# ----------------------
data = poll_mqtt_message()
if data:
    temp = float(data.get("temp"))
    hum = float(data.get("hum"))

    pred = model.predict([[temp, hum]])[0]

    row = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "temp": temp,
        "hum": hum,
        "pred": pred
    }
    st.session_state.messages.append(row)

    # send output to ESP32
    if pred == "Panas":
        client.publish(TOPIC_OUTPUT, "ALERT_ON")
    else:
        client.publish(TOPIC_OUTPUT, "ALERT_OFF")


# ----------------------
# UI
# ----------------------
st.title("🔥 IoT ML Realtime Dashboard (Cloud Mode)")

st.subheader("MQTT Connection")
st.write("Broker:", BROKER)
st.write("Status:", "Connected")

st.subheader("Latest Data")
if st.session_state.messages:
    st.write(st.session_state.messages[-1])
else:
    st.info("Waiting for data...")

st.subheader("Realtime Log")
df = pd.DataFrame(st.session_state.messages)
st.dataframe(df.tail(20))

# Auto-refresh every 1 second
st.experimental_rerun()
