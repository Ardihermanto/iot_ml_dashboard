import streamlit as st
import pandas as pd
import json
import joblib
import time
from datetime import datetime
import paho.mqtt.client as mqtt

st.set_page_config(page_title="IoT ML Dashboard", layout="wide")
st.title("🔥 IoT ML Dashboard — Stable No-Thread Version")

# -------------------------------
# CONFIG
# -------------------------------
MQTT_BROKER = st.secrets.get("MQTT_BROKER", "broker.hivemq.com")
MQTT_PORT = int(st.secrets.get("MQTT_PORT", 1883))
TOPIC_SENSOR = st.secrets.get("TOPIC_SENSOR", "iot/class/session5/sensor")
TOPIC_OUTPUT = st.secrets.get("TOPIC_OUTPUT", "iot/class/session5/output")
MODEL_PATH = st.secrets.get("MODEL_PATH", "iot_temp_model.pkl")

# -------------------------------
# SESSION STATE
# -------------------------------
if "logs" not in st.session_state:
    st.session_state.logs = []
if "last" not in st.session_state:
    st.session_state.last = None
if "mqtt_connected" not in st.session_state:
    st.session_state.mqtt_connected = False

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Gagal load model: {e}")
        return None

model = load_model()

# -------------------------------
# MQTT SETUP (NO CALLBACK)
# POLLING-BASED (AMAN)
# -------------------------------
client = mqtt.Client()

try:
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    st.session_state.mqtt_connected = True
except Exception as e:
    st.session_state.mqtt_connected = False
    st.error(f"MQTT connect error: {e}")

client.subscribe(TOPIC_SENSOR)

# -------------------------------
# POLLING MESSAGE LOOP
# -------------------------------
def read_messages():
    try:
        rc = client.loop(timeout=0.05)
        if rc != mqtt.MQTT_ERR_SUCCESS:
            st.session_state.mqtt_connected = False
    except:
        st.session_state.mqtt_connected = False

    msg = client.simple_read()
    if msg is None:
        return None

    try:
        data = json.loads(msg.payload.decode())
        return data
    except:
        return None


# -------------------------------
# MAIN LOOP
# -------------------------------
placeholder_status = st.empty()
placeholder_last = st.empty()
placeholder_chart = st.empty()
placeholder_logs = st.empty()

while True:
    data = read_messages()

    if data:
        temp = float(data["temp"])
        hum = float(data["hum"])
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # prediction
        pred = model.predict([[temp, hum]])[0]

        # save
        row = {"ts": ts, "temp": temp, "hum": hum, "pred": pred}
        st.session_state.last = row
        st.session_state.logs.append(row)

        # send output
        if pred == "Panas":
            client.publish(TOPIC_OUTPUT, "ALERT_ON")
        else:
            client.publish(TOPIC_OUTPUT, "ALERT_OFF")

    # ==== UI Update ====
    placeholder_status.write(f"**MQTT Connected:** {st.session_state.mqtt_connected}")

    if st.session_state.last:
        placeholder_last.write(st.session_state.last)
    else:
        placeholder_last.info("Waiting for data...")

    df = pd.DataFrame(st.session_state.logs)
    if len(df) > 0:
        placeholder_chart.line_chart(df[["temp", "hum"]].tail(200))
        placeholder_logs.dataframe(df.iloc[::-1].head(20))
    else:
        placeholder_chart.write("No data yet")

    time.sleep(1)
