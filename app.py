# ============================================
#  STREAMLIT IoT ML DASHBOARD (FINAL - WSS)
#  MQTT via WSS 8084 for Streamlit Cloud
#  ESP32 stays on TCP 1883 (no change needed)
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import asyncio
import threading
import queue
from datetime import datetime
import plotly.graph_objs as go
from asyncio_mqtt import Client, MqttError

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="IoT ML Dashboard", layout="wide")
st.title("IoT ML Realtime Dashboard")

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("iot_temp_model.pkl")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model()

# -----------------------------
# MQTT CONFIG (Streamlit Cloud uses WSS)
# -----------------------------
BROKER_URL = st.secrets.get("MQTT_BROKER", "broker.emqx.io")
BROKER_PORT = int(st.secrets.get("MQTT_PORT", 8084))  # WSS for Streamlit Cloud
TOPIC_SENSOR = st.secrets.get("TOPIC_SENSOR", "iot/class/session5/sensor")
TOPIC_OUTPUT = st.secrets.get("TOPIC_OUTPUT", "iot/class/session5/output")


# -----------------------------
# QUEUE for passing MQTT → UI safely
# -----------------------------
mqtt_queue = queue.Queue()


# -----------------------------
# ASYNC MQTT LOOP
# -----------------------------
async def mqtt_handler():
    """Async MQTT listener (WSS)"""
    try:
        async with Client(
            BROKER_URL,
            port=BROKER_PORT,
            username=None,
            password=None,
            tls_context=None,
            transport="websockets",   # 🔥 IMPORTANT FOR STREAMLIT CLOUD
        ) as client:

            await client.subscribe(TOPIC_SENSOR)
            st.session_state.connected = True

            async with client.unfiltered_messages() as messages:
                async for msg in messages:
                    try:
                        payload = json.loads(msg.payload.decode())
                        temp = float(payload.get("temp"))
                        hum = float(payload.get("hum"))
                        ts = datetime.utcnow().isoformat()

                        # ML prediction
                        pred = "N/A"
                        conf = None
                        if model:
                            X = [[temp, hum]]
                            pred = model.predict(X)[0]
                            try:
                                conf = float(np.max(model.predict_proba(X)))
                            except:
                                conf = None

                        mqtt_queue.put({
                            "ts": ts,
                            "temp": temp,
                            "hum": hum,
                            "pred": pred,
                            "conf": conf
                        })

                        # SEND BACK to ESP32
                        if pred == "Panas":
                            await client.publish(TOPIC_OUTPUT, "ALERT_ON")
                        else:
                            await client.publish(TOPIC_OUTPUT, "ALERT_OFF")

                    except Exception as e:
                        print("Error processing message:", e)

    except MqttError as e:
        st.session_state.connected = False
        print("MQTT error:", e)


# -----------------------------
# START BACKGROUND ASYNC THREAD
# -----------------------------
def start_async_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(mqtt_handler())

if "async_thread_started" not in st.session_state:
    t = threading.Thread(target=start_async_loop, daemon=True)
    t.start()
    st.session_state.async_thread_started = True


# -----------------------------
# SESSION STATE
# -----------------------------
if "logs" not in st.session_state:
    st.session_state.logs = []

if "connected" not in st.session_state:
    st.session_state.connected = False


# -----------------------------
# PROCESS QUEUE (MQTT → UI)
# -----------------------------
while not mqtt_queue.empty():
    msg = mqtt_queue.get()
    st.session_state.logs.append(msg)


# -----------------------------
# UI LAYOUT
# -----------------------------
left, right = st.columns([1, 2])

# LEFT PANEL
with left:
    st.subheader("Connection Status")
    st.metric("MQTT Broker", BROKER_URL)
    st.metric("MQTT Connected", "YES" if st.session_state.connected else "NO")

    if st.session_state.logs:
        st.write("**Last Reading**")
        st.write(pd.DataFrame([st.session_state.logs[-1]]).T)
    else:
        st.info("Waiting for sensor data...")

    if st.button("Save logs to CSV"):
        df = pd.DataFrame(st.session_state.logs)
        fn = f"iot_log_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
        st.download_button("Download CSV", df.to_csv(index=False), file_name=fn)


# RIGHT PANEL
with right:
    st.subheader("Realtime Chart")

    df = pd.DataFrame(st.session_state.logs)
    if not df.empty:
        df = df.tail(200)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["ts"], y=df["temp"], mode="lines+markers", name="Temperature"))
        fig.add_trace(go.Scatter(x=df["ts"], y=df["hum"], mode="lines+markers", name="Humidity"))

        # color markers by prediction class
        colors = df["pred"].map({
            "Panas": "red",
            "Normal": "green",
            "Dingin": "blue"
        }).fillna("white")

        fig.update_traces(marker=dict(size=10, color=colors), selector=dict(mode="markers"))
        fig.update_layout(height=450)

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recent Data")
    if st.session_state.logs:
        st.dataframe(df.iloc[::-1].head(20))
    else:
        st.write("—")


st.markdown("---")
st.caption("IoT ML Dashboard — Powered by Streamlit + EMQX WSS")
