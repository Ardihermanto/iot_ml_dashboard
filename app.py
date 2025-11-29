import streamlit as st
import asyncio
from asyncio_mqtt import Client
import pandas as pd
import numpy as np
import json
import joblib
from datetime import datetime

st.set_page_config(page_title="IoT MQTT Dashboard", layout="wide")
st.title("🔥 IoT ML Dashboard — Asyncio Version (Stable)")

# -----------------------------
# CONFIG
# -----------------------------
MQTT_BROKER = st.secrets.get("MQTT_BROKER", "broker.emqx.io")
MQTT_PORT = int(st.secrets.get("MQTT_PORT", 1883))
TOPIC_SENSOR = st.secrets.get("TOPIC_SENSOR", "iot/class/session5/sensor")
TOPIC_OUTPUT = st.secrets.get("TOPIC_OUTPUT", "iot/class/session5/output")
MODEL_PATH = st.secrets.get("MODEL_PATH", "iot_temp_model.pkl")

# -----------------------------
# STATE
# -----------------------------
if "logs" not in st.session_state:
    st.session_state.logs = []
if "last" not in st.session_state:
    st.session_state.last = None
if "connected" not in st.session_state:
    st.session_state.connected = False

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error("Model load error: " + str(e))
        return None

model = load_model()

# -----------------------------
# MQTT LISTENER (ASYNC)
# -----------------------------
async def mqtt_listener():
    try:
        async with Client(MQTT_BROKER, port=MQTT_PORT) as client:
            st.session_state.connected = True
            await client.subscribe(TOPIC_SENSOR)

            async with client.messages() as messages:
                async for msg in messages:
                    try:
                        data = json.loads(msg.payload.decode())
                        temp = float(data["temp"])
                        hum = float(data["hum"])
                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        # prediction
                        pred = model.predict([[temp, hum]])[0]

                        row = {"ts": ts, "temp": temp, "hum": hum, "pred": pred}
                        st.session_state.last = row
                        st.session_state.logs.append(row)

                        # send output
                        if pred == "Panas":
                            await client.publish(TOPIC_OUTPUT, "ALERT_ON")
                        else:
                            await client.publish(TOPIC_OUTPUT, "ALERT_OFF")

                    except Exception as e:
                        print("Parse error:", e)

    except Exception as e:
        st.session_state.connected = False
        print("MQTT error:", e)
        await asyncio.sleep(2)
        await mqtt_listener()   # auto reconnect


# -----------------------------
# START LISTENER (ONCE)
# -----------------------------
if "listener_started" not in st.session_state:
    st.session_state.listener_started = True
    asyncio.create_task(mqtt_listener())


# -----------------------------
# UI DISPLAY
# -----------------------------
st.subheader("MQTT Status")
st.write("Connected:", st.session_state.connected)

st.subheader("Last Data")
if st.session_state.last:
    st.write(st.session_state.last)
else:
    st.info("Waiting for data...")

st.subheader("Live Table")
if st.session_state.logs:
    df = pd.DataFrame(st.session_state.logs)
    st.dataframe(df.tail(20))
else:
    st.write("No data yet")
