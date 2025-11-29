import streamlit as st
import pandas as pd
import json
import joblib
import time
from datetime import datetime
import paho.mqtt.client as mqtt

# -------------------
# PAGE CONFIG
# -------------------
st.set_page_config(page_title="IoT ML Dashboard", layout="wide")
st.title("🔥 IoT ML Dashboard — Stable MQTT (No Thread)")

# -------------------
# CONFIG
# -------------------
MQTT_BROKER = st.secrets.get("MQTT_BROKER", "broker.hivemq.com")
MQTT_PORT = int(st.secrets.get("MQTT_PORT", 1883))
TOPIC_SENSOR = st.secrets.get("TOPIC_SENSOR", "iot/class/session5/sensor")
TOPIC_OUTPUT = st.secrets.get("TOPIC_OUTPUT", "iot/class/session5/output")
MODEL_PATH = st.secrets.get("MODEL_PATH", "iot_temp_model.pkl")

# -------------------
# SESSION STATE
# -------------------
if "logs" not in st.session_state:
    st.session_state.logs = []
if "last" not in st.session_state:
    st.session_state.last = None
if "mqtt_connected" not in st.session_state:
    st.session_state.mqtt_connected = False

# -------------------
# LOAD MODEL
# -------------------
@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model(MODEL_PATH)

# -------------------
# MQTT CLIENT (no callbacks)
# -------------------
client = mqtt.Client()

try:
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    st.session_state.mqtt_connected = True
except Exception as e:
    st.session_state.mqtt_connected = False
    st.error(f"MQTT connect error: {e}")

client.subscribe(TOPIC_SENSOR)

# -------------------
# UI placeholders
# -------------------
ph_status = st.empty()
ph_last = st.empty()
ph_chart = st.empty()
ph_logs = st.empty()

# -------------------
# MAIN LOOP
# -------------------
while True:
    rc = client.loop(timeout=0.1)  # process incoming MQTT

    # check message manually
    msg = client._sock_recv() if hasattr(client, "_sock_recv") else None

    if msg:
        try:
            payload = msg[1].payload.decode()
            data = json.loads(payload)
        except:
            data = None

        if data:
            temp = float(data["temp"])
            hum = float(data["hum"])
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # ML prediction
            pred = model.predict([[temp, hum]])[0]

            row = {"ts": ts, "temp": temp, "hum": hum, "pred": pred}
            st.session_state.last = row
            st.session_state.logs.append(row)

            # send back to ESP32
            if pred == "Panas":
                client.publish(TOPIC_OUTPUT, "ALERT_ON")
            else:
                client.publish(TOPIC_OUTPUT, "ALERT_OFF")

    # -------------------
    # UPDATE UI
    # -------------------
    ph_status.write(f"**MQTT Connected:** {st.session_state.mqtt_connected}")

    if st.session_state.last:
        ph_last.write(st.session_state.last)
    else:
        ph_last.info("Waiting for data...")

    df = pd.DataFrame(st.session_state.logs)

    if not df.empty:
        ph_chart.line_chart(df[["temp", "hum"]].tail(200))
        ph_logs.dataframe(df.iloc[::-1].head(20))
    else:
        ph_chart.write("No data yet")

    time.sleep(1)
