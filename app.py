###############################################################
# IOT ML REALTIME DASHBOARD (STREAMLIT SAFE VERSION)
# NO THREADS, NO asyncio.run(), NO SCRIPT CONTEXT ERRORS
###############################################################

import streamlit as st
st.set_page_config(page_title="IoT ML Realtime Dashboard", layout="wide")

import asyncio
import json
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import plotly.graph_objs as go
from asyncio_mqtt import Client, MqttError

###############################################################
# CONFIG
###############################################################

MQTT_BROKER = st.secrets.get("MQTT_BROKER", "broker.emqx.io")
MQTT_PORT   = int(st.secrets.get("MQTT_PORT", 1883))
TOPIC_SENSOR = st.secrets.get("TOPIC_SENSOR", "iot/class/session5/sensor")
TOPIC_OUTPUT = st.secrets.get("TOPIC_OUTPUT", "iot/class/session5/output")
MODEL_PATH = st.secrets.get("MODEL_PATH", "iot_temp_model.pkl")

###############################################################
# LOAD ML MODEL
###############################################################

@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error("Failed to load model: " + str(e))
        return None

model = load_model(MODEL_PATH)

###############################################################
# INIT SESSION VARIABLES
###############################################################

if "logs" not in st.session_state:
    st.session_state.logs = []

if "last" not in st.session_state:
    st.session_state.last = None

if "mqtt_connected" not in st.session_state:
    st.session_state.mqtt_connected = False

if "listener_started" not in st.session_state:
    st.session_state.listener_started = False

###############################################################
# ASYNC MQTT LISTENER
###############################################################

async def mqtt_listener():
    """ Runs forever and listens to MQTT messages safely """
    while True:
        try:
            async with Client(MQTT_BROKER, port=MQTT_PORT) as client:
                st.session_state.mqtt_connected = True
                await client.subscribe(TOPIC_SENSOR)

                async with client.unfiltered_messages() as messages:
                    async for msg in messages:
                        try:
                            data = json.loads(msg.payload.decode())
                            temp = float(data["temp"])
                            hum  = float(data["hum"])
                        except:
                            continue

                        ts = datetime.utcnow().isoformat()

                        # ML Prediction
                        pred = "N/A"
                        conf = None
                        try:
                            X = [[temp, hum]]
                            pred = model.predict(X)[0]
                            try:
                                conf = float(np.max(model.predict_proba(X)))
                            except:
                                conf = None
                        except:
                            pred = "ERR"

                        # Add log
                        row = {
                            "ts": ts,
                            "temp": temp,
                            "hum": hum,
                            "pred": pred,
                            "conf": conf
                        }
                        st.session_state.logs.append(row)
                        st.session_state.last = row

                        # Publish ML output back to ESP32
                        if pred == "Panas":
                            await client.publish(TOPIC_OUTPUT, "ALERT_ON")
                        else:
                            await client.publish(TOPIC_OUTPUT, "ALERT_OFF")

        except MqttError:
            st.session_state.mqtt_connected = False
            await asyncio.sleep(3)  # reconnect loop


###############################################################
# START MQTT LISTENER SAFELY (NO THREAD)
###############################################################

async def start_listener_once():
    if not st.session_state.listener_started:
        st.session_state.listener_started = True
        asyncio.create_task(mqtt_listener())

asyncio.create_task(start_listener_once())

###############################################################
# UI LAYOUT
###############################################################

st.title("IoT ML Realtime Dashboard")

col1, col2 = st.columns([1, 2])

###############################################################
# LEFT PANEL
###############################################################

with col1:
    st.subheader("Connection Status")
    st.metric("MQTT Broker", MQTT_BROKER)
    st.metric("MQTT Connected", "Yes" if st.session_state.mqtt_connected else "No")

    st.subheader("Last Reading")
    if st.session_state.last:
        st.write(pd.DataFrame([st.session_state.last]).T)
    else:
        st.info("Waiting for sensor data...")

    st.subheader("Save Log")
    if st.button("Download CSV Log"):
        df = pd.DataFrame(st.session_state.logs)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download",
            csv,
            "iot_realtime_log.csv",
            "text/csv",
            key="download-csv"
        )

###############################################################
# RIGHT PANEL
###############################################################

with col2:
    st.subheader("Live Chart")

    df = pd.DataFrame(st.session_state.logs)
    if not df.empty:
        df_plot = df.tail(300)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_plot["ts"], y=df_plot["temp"],
            mode="lines+markers", name="Temperature (°C)"
        ))
        fig.add_trace(go.Scatter(
            x=df_plot["ts"], y=df_plot["hum"],
            mode="lines+markers", name="Humidity (%)"
        ))

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data yet")

    st.subheader("Recent Readings")
    if len(st.session_state.logs) > 0:
        st.dataframe(pd.DataFrame(st.session_state.logs).iloc[::-1].head(20))
    else:
        st.write("—")

###############################################################
# FOOTER
###############################################################

st.markdown("---")
st.write("Total messages:", len(st.session_state.logs))
