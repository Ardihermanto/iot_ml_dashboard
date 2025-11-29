# app.py
# MQTT over WebSocket (WSS) + Streamlit safe integration
# - Background thread runs paho-mqtt (transport="websockets")
# - Background thread NEVER calls Streamlit APIs
# - Two queues: mqtt_queue (inbound), publish_queue (outbound)
# - Main Streamlit thread consumes mqtt_queue, does ML predict, updates UI,
#   and enqueues publish messages to publish_queue.

import streamlit as st
st.set_page_config(page_title="IoT ML Dashboard (WSS)", layout="wide")

import pandas as pd
import numpy as np
import joblib
import json
import queue
import threading
import time
from datetime import datetime
import plotly.graph_objs as go
import paho.mqtt.client as mqtt
import ssl

# --------------------
# CONFIG (load in main thread only)
# --------------------
# Default broker: broker.emqx.io supports websocket secure on port 8084 path /mqtt
MQTT_BROKER = st.secrets.get("MQTT_BROKER", "broker.emqx.io")
MQTT_PORT   = int(st.secrets.get("MQTT_PORT", 8084))  # wss port for EMQX
MQTT_PATH   = st.secrets.get("MQTT_PATH", "/mqtt")   # websocket path
TOPIC_SENSOR = st.secrets.get("TOPIC_SENSOR", "iot/class/session5/sensor")
TOPIC_OUTPUT = st.secrets.get("TOPIC_OUTPUT", "iot/class/session5/output")
MODEL_PATH = st.secrets.get("MODEL_PATH", "iot_temp_model.pkl")
CLIENT_ID = st.secrets.get("CLIENT_ID", "streamlit_iot_client")

# --------------------
# QUEUES (thread-safe)
# --------------------
mqtt_queue = queue.Queue()    # inbound messages (from broker -> main thread)
publish_queue = queue.Queue() # outbound messages (from main thread -> broker)

# --------------------
# Session state init
# --------------------
if "logs" not in st.session_state:
    st.session_state.logs = []
if "last" not in st.session_state:
    st.session_state.last = None
if "override" not in st.session_state:
    st.session_state.override = None
if "mqtt_connected" not in st.session_state:
    st.session_state.mqtt_connected = False
if "mqtt_thread_started" not in st.session_state:
    st.session_state.mqtt_thread_started = False

# --------------------
# Load ML model
# --------------------
@st.cache_resource
def load_model(path):
    return joblib.load(path)

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# --------------------
# MQTT callbacks (NO Streamlit calls here!)
# --------------------
def _on_connect(client, userdata, flags, rc):
    # Put system status into queue (main thread will pick it)
    if rc == 0:
        mqtt_queue.put({"system": "connected"})
        # subscribe
        try:
            client.subscribe(TOPIC_SENSOR)
        except Exception:
            pass
    else:
        mqtt_queue.put({"system": "disconnected"})

def _on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode()
        data = json.loads(payload)
        temp = float(data.get("temp"))
        hum = float(data.get("hum"))
        ts = datetime.utcnow().isoformat()
        mqtt_queue.put({"ts": ts, "temp": temp, "hum": hum})
    except Exception:
        # ignore malformed messages
        return

# --------------------
# MQTT background thread: uses websockets transport + TLS
# --------------------
def mqtt_background_worker():
    """
    Background worker that:
    - connects via websockets (WSS) to broker
    - subscribes to TOPIC_SENSOR
    - pushes inbound messages to mqtt_queue
    - pulls publish_queue and publishes messages
    This thread MUST NOT call Streamlit APIs.
    """
    while True:
        try:
            # create client with websockets transport
            client = mqtt.Client(client_id=CLIENT_ID, transport="websockets")

            # optional: set username/password if provided in secrets (read only in main thread)
            mqtt_user = st.secrets.get("MQTT_USER")
            mqtt_pass = st.secrets.get("MQTT_PASS")
            if mqtt_user and mqtt_pass:
                client.username_pw_set(mqtt_user, mqtt_pass)

            # set TLS (for wss): use system default CA certs
            try:
                client.tls_set(cert_reqs=ssl.CERT_REQUIRED, tls_version=ssl.PROTOCOL_TLS_CLIENT)
            except Exception:
                # fallback: try tls_set without params
                try:
                    client.tls_set()
                except Exception:
                    pass

            # set websocket path (some brokers use /mqtt)
            try:
                client.ws_set_options(path=MQTT_PATH)
            except Exception:
                pass

            client.on_connect = _on_connect
            client.on_message = _on_message

            # connect (host, port)
            client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)

            # start network loop in separate paho thread
            client.loop_start()

            # process publish_queue non-blocking
            while True:
                # publish outgoing messages if any
                try:
                    pkt = publish_queue.get_nowait()
                except queue.Empty:
                    pkt = None

                if pkt:
                    tpc = pkt.get("topic")
                    payload = pkt.get("payload")
                    qos = pkt.get("qos", 0)
                    retain = pkt.get("retain", False)
                    try:
                        client.publish(tpc, payload, qos=qos, retain=retain)
                    except Exception:
                        # ignore publish error; main thread can retry later
                        pass

                time.sleep(0.1)  # small sleep to yield
        except Exception as ex:
            # signal disconnect to main thread, wait then retry
            try:
                mqtt_queue.put({"system": "disconnected"})
            except Exception:
                pass
            time.sleep(3)
            continue

# start background thread once
if not st.session_state.mqtt_thread_started:
    threading.Thread(target=mqtt_background_worker, daemon=True).start()
    st.session_state.mqtt_thread_started = True
    # give thread a moment
    time.sleep(0.2)

# --------------------
# Main thread: consume mqtt_queue (safe)
# --------------------
while not mqtt_queue.empty():
    item = mqtt_queue.get()
    # system message (connected/disconnected)
    if "system" in item:
        st.session_state.mqtt_connected = (item["system"] == "connected")
        continue

    # sensor message
    ts = item.get("ts")
    temp = item.get("temp")
    hum = item.get("hum")

    # prediction in main thread (safe)
    try:
        pred = model.predict([[temp, hum]])[0]
    except Exception:
        pred = "ERR"

    row = {"ts": ts, "temp": temp, "hum": hum, "pred": pred}
    st.session_state.logs.append(row)
    st.session_state.last = row

    # enqueue automatic output to publish_queue (main->bg thread)
    if st.session_state.override is None:
        out_msg = "ALERT_ON" if pred == "Panas" else "ALERT_OFF"
        publish_queue.put({"topic": TOPIC_OUTPUT, "payload": out_msg})

# --------------------
# UI
# --------------------
st.title("IoT ML Realtime Dashboard — WSS MQTT")

left, right = st.columns([1,2])

with left:
    st.subheader("Status")
    st.metric("MQTT Broker", MQTT_BROKER)
    st.metric("MQTT Connected", "Yes" if st.session_state.mqtt_connected else "No")

    st.subheader("Manual Override")
    c1, c2 = st.columns(2)
    if c1.button("Force ALERT ON"):
        st.session_state.override = "ON"
        # enqueue manual publish (main -> background)
        publish_queue.put({"topic": TOPIC_OUTPUT, "payload": "ALERT_ON"})
        st.success("Enqueued ALERT_ON")
    if c2.button("Force ALERT OFF"):
        st.session_state.override = "OFF"
        publish_queue.put({"topic": TOPIC_OUTPUT, "payload": "ALERT_OFF"})
        st.success("Enqueued ALERT_OFF")
    if st.button("Clear Override"):
        st.session_state.override = None
        st.info("Auto alerts resumed")

    st.subheader("Last Reading")
    if st.session_state.last:
        st.write(pd.DataFrame([st.session_state.last]).T)
    else:
        st.info("No data yet")

    if st.button("Save Log to CSV"):
        if st.session_state.logs:
            df_export = pd.DataFrame(st.session_state.logs)
            fn = "iot_log_" + datetime.utcnow().strftime("%Y%m%d%H%M%S") + ".csv"
            st.download_button("Download CSV", df_export.to_csv(index=False).encode("utf-8"), file_name=fn)
        else:
            st.warning("No logs to save")

with right:
    st.subheader("Live Chart")
    df = pd.DataFrame(st.session_state.logs)
    if not df.empty:
        df = df.tail(200)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["ts"], y=df["temp"], mode="lines+markers", name="Temp (°C)"))
        fig.add_trace(go.Scatter(x=df["ts"], y=df["hum"], mode="lines+markers", name="Hum (%)"))
        if "pred" in df.columns:
            colors = ["red" if p=="Panas" else "green" if p=="Normal" else "blue" for p in df["pred"]]
            fig.update_traces(marker=dict(color=colors), selector=dict(mode="markers"))
        fig.update_layout(xaxis_title="timestamp", height=500)
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
