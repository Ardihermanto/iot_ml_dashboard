# app.py — Asyncio-MQTT version for Streamlit (safe)
import streamlit as st
st.set_page_config(page_title="IoT ML Realtime Dashboard (Async MQTT)", layout="wide")

import pandas as pd
import numpy as np
import joblib
import json
import queue
import threading
import time
from datetime import datetime
import plotly.graph_objs as go

# Note: asyncio-mqtt used in background thread
# background thread will NOT touch Streamlit APIs
# it only puts data into mqtt_queue
# main thread (Streamlit) will consume mqtt_queue and update UI

# Read config in MAIN thread only (no st.* inside background thread)
CONFIG = {
    "MQTT_BROKER": st.secrets.get("MQTT_BROKER", "broker.hivemq.com"),
    "MQTT_PORT": int(st.secrets.get("MQTT_PORT", 1883)),
    "TOPIC_SENSOR": st.secrets.get("TOPIC_SENSOR", "iot/class/session5/sensor"),
    "TOPIC_OUTPUT": st.secrets.get("TOPIC_OUTPUT", "iot/class/session5/output"),
    "MODEL_PATH": st.secrets.get("MODEL_PATH", "iot_temp_model.pkl"),
    "CLIENT_ID": st.secrets.get("CLIENT_ID", "streamlit_iot_client")
}

MQTT_BROKER = CONFIG["MQTT_BROKER"]
MQTT_PORT = CONFIG["MQTT_PORT"]
TOPIC_SENSOR = CONFIG["TOPIC_SENSOR"]
TOPIC_OUTPUT = CONFIG["TOPIC_OUTPUT"]
MODEL_PATH = CONFIG["MODEL_PATH"]
CLIENT_ID = CONFIG["CLIENT_ID"]

# thread-safe queue (producer: mqtt thread, consumer: main thread)
mqtt_queue = queue.Queue()

# Streamlit session init
if "logs" not in st.session_state:
    st.session_state.logs = []
if "last" not in st.session_state:
    st.session_state.last = None
if "override" not in st.session_state:
    st.session_state.override = None
if "mqtt_client" not in st.session_state:
    st.session_state.mqtt_client = None
if "mqtt_connected" not in st.session_state:
    st.session_state.mqtt_connected = False

# Load ML model (main thread)
@st.cache_resource
def load_model(path):
    return joblib.load(path)

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# ---------------------------
# Background async MQTT worker
# ---------------------------
def mqtt_async_thread():
    """
    Runs an asyncio-mqtt client inside a background thread.
    The thread NEVER calls Streamlit APIs. It only puts dict messages
    into mqtt_queue for the main thread to consume.
    """
    import asyncio
    from asyncio_mqtt import Client, MqttError

    async def runner():
        while True:
            try:
                async with Client(MQTT_BROKER, port=MQTT_PORT, client_id=CLIENT_ID) as client:
                    # notify main thread about connection
                    mqtt_queue.put({"system": "connected"})
                    # subscribe
                    await client.subscribe(TOPIC_SENSOR)
                    # consume messages
                    async with client.unfiltered_messages() as messages:
                        async for message in messages:
                            try:
                                payload = message.payload.decode()
                                data = json.loads(payload)
                                temp = float(data.get("temp"))
                                hum = float(data.get("hum"))
                                ts = datetime.utcnow().isoformat()
                                mqtt_queue.put({
                                    "ts": ts,
                                    "temp": temp,
                                    "hum": hum
                                })
                            except Exception:
                                # ignore bad messages
                                continue
            except MqttError as me:
                # signal disconnect to main thread and retry after backoff
                mqtt_queue.put({"system": "disconnected"})
                await asyncio.sleep(3)
            except Exception:
                mqtt_queue.put({"system": "disconnected"})
                await asyncio.sleep(3)

    # Run asyncio event loop in this thread
    try:
        asyncio.run(runner())
    except Exception:
        # if thread main loop exits, signal disconnected
        mqtt_queue.put({"system": "disconnected"})

# start background thread once (safe — thread does NOT touch st)
if "mqtt_thread_started" not in st.session_state:
    t = threading.Thread(target=mqtt_async_thread, daemon=True)
    t.start()
    st.session_state.mqtt_thread_started = True
    # small pause to let thread attempt connect (optional)
    time.sleep(0.2)

# ---------------------------
# Main thread: consume queue
# ---------------------------
# Process all available messages in the queue (non-blocking)
while not mqtt_queue.empty():
    item = mqtt_queue.get()
    # system messages
    if "system" in item:
        st.session_state.mqtt_connected = (item["system"] == "connected")
        continue

    # sensor message
    ts = item.get("ts")
    temp = item.get("temp")
    hum = item.get("hum")

    # safe: prediction in main thread
    try:
        pred = model.predict([[temp, hum]])[0]
    except Exception as e:
        pred = "ERR"

    row = {"ts": ts, "temp": temp, "hum": hum, "pred": pred}
    st.session_state.logs.append(row)
    st.session_state.last = row

    # publish back to ESP32 if mqtt client object exists (we store a small helper wrapper)
    # Note: we do NOT call MQTT connect/publish from background thread anymore;
    # instead we try to create a minimal publish helper using asyncio-mqtt in a short sync wrapper.
    # For Streamlit Cloud we prefer to publish via a short synchronous attempt (best-effort).
    try:
        # publish using a short-lived asyncio client to send the ALERT message
        if st.session_state.override is None:
            # choose message
            out_msg = "ALERT_ON" if pred == "Panas" else "ALERT_OFF"
            # use asyncio to publish in a blocking manner (short-lived client)
            # this keeps background thread responsibilities separate
            import asyncio
            from asyncio_mqtt import Client as AsyncClient

            async def pub_once():
                try:
                    async with AsyncClient(MQTT_BROKER, port=MQTT_PORT) as pub_client:
                        await pub_client.publish(TOPIC_OUTPUT, out_msg)
                except Exception:
                    pass

            try:
                asyncio.run(pub_once())
            except Exception:
                # If asyncio.run fails inside Streamlit environment, ignore publish
                pass
    except Exception:
        pass

# ---------------------------
# UI
# ---------------------------
st.title("IoT ML Realtime Dashboard — Async MQTT")

left, right = st.columns([1, 2])

with left:
    st.subheader("Status")
    st.metric("MQTT Broker", MQTT_BROKER)
    st.metric("MQTT Connected", "Yes" if st.session_state.mqtt_connected else "No")

    st.subheader("Manual Override")
    c1, c2 = st.columns(2)
    if c1.button("Force ALERT ON"):
        st.session_state.override = "ON"
        # publish manual override message
        try:
            import asyncio
            from asyncio_mqtt import Client as AsyncClient
            async def pub_on():
                async with AsyncClient(MQTT_BROKER, port=MQTT_PORT) as client:
                    await client.publish(TOPIC_OUTPUT, "ALERT_ON")
            asyncio.run(pub_on())
        except Exception:
            st.warning("Publish failed (best-effort).")
    if c2.button("Force ALERT OFF"):
        st.session_state.override = "OFF"
        try:
            import asyncio
            from asyncio_mqtt import Client as AsyncClient
            async def pub_off():
                async with AsyncClient(MQTT_BROKER, port=MQTT_PORT) as client:
                    await client.publish(TOPIC_OUTPUT, "ALERT_OFF")
            asyncio.run(pub_off())
        except Exception:
            st.warning("Publish failed (best-effort).")
    if st.button("Clear Override"):
        st.session_state.override = None
        st.success("Override cleared")

    st.subheader("Last Reading")
    if st.session_state.last:
        st.write(pd.DataFrame([st.session_state.last]).T)
    else:
        st.info("No data yet")

with right:
    st.subheader("Live Chart")
    df = pd.DataFrame(st.session_state.logs)
    if not df.empty:
        df = df.tail(200)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["ts"], y=df["temp"], mode="lines+markers", name="Temp (°C)"))
        fig.add_trace(go.Scatter(x=df["ts"], y=df["hum"], mode="lines+markers", name="Hum (%)"))
        # color markers by prediction if exists
        if "pred" in df.columns:
            colors = ["red" if p == "Panas" else "green" if p == "Normal" else "blue" for p in df["pred"]]
            fig.update_traces(marker=dict(color=colors), selector=dict(mode="markers"))
        fig.update_layout(xaxis_title="timestamp", height=450)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data yet")

    st.subheader("Recent Data")
    if st.session_state.logs:
        st.dataframe(pd.DataFrame(st.session_state.logs).iloc[::-1].head(20))
    else:
        st.write("—")

st.markdown("---")
st.write("Manual override:", st.session_state.override)
st.write("Total messages:", len(st.session_state.logs))
