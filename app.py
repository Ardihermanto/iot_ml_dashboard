import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import json
import joblib
from datetime import datetime
from paho.mqtt.client import Client as MQTTClient, MQTTv5
import plotly.graph_objs as go

# -----------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------
st.set_page_config(page_title="IoT ML Realtime Dashboard", layout="wide")

# -----------------------------------------------------------
# MQTT CONFIG (ambil dari secrets atau default)
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
        st.error(f"❌ Model tidak bisa diload: {e}")
        return None

model = load_model(MODEL_PATH)

# -----------------------------------------------------------
# SESSION STATE
# -----------------------------------------------------------
if "mqtt_connected" not in st.session_state:
    st.session_state.mqtt_connected = False

if "logs" not in st.session_state:
    st.session_state.logs = []

if "last" not in st.session_state:
    st.session_state.last = None

if "override" not in st.session_state:
    st.session_state.override = None  # "ON" / "OFF" / None

# -----------------------------------------------------------
# MQTT CLIENT (NO THREAD, NO DEPRECATION)
# Using protocol MQTTv5 + callback_api_version=5
# -----------------------------------------------------------
mqtt_client = MQTTClient(
    protocol=MQTTv5,
    callback_api_version=5,
    client_id="streamlit_iot_dashboard"
)

# -----------------------------------------------------------
# CALLBACK: CONNECT
# -----------------------------------------------------------
def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        st.session_state.mqtt_connected = True
        client.subscribe(TOPIC_SENSOR)
        print("MQTT Connected & Subscribed:", TOPIC_SENSOR)
    else:
        print("MQTT connect failed, rc=", rc)
        st.session_state.mqtt_connected = False

mqtt_client.on_connect = on_connect

# -----------------------------------------------------------
# CALLBACK: MESSAGE
# -----------------------------------------------------------
def on_message(client, userdata, msg, properties=None):
    try:
        data = json.loads(msg.payload.decode())
        temp = float(data["temp"])
        hum = float(data["hum"])
    except:
        print("Parsing error:", msg.payload)
        return

    ts = datetime.utcnow().isoformat()

    # PREDIKSI ML
    pred = "N/A"
    conf = None

    if model is not None:
        try:
            X = [[temp, hum]]
            pred = model.predict(X)[0]
            try:
                conf = float(np.max(model.predict_proba(X)))
            except:
                conf = None
        except:
            pred = "ERR"

    row = {"ts": ts, "temp": temp, "hum": hum, "pred": pred, "conf": conf}
    st.session_state.logs.append(row)
    st.session_state.last = row

    # AUTO-ALERT → KIRIM KE ESP32
    if st.session_state.override is None:
        if pred == "Panas":
            client.publish(TOPIC_OUTPUT, "ALERT_ON")
        else:
            client.publish(TOPIC_OUTPUT, "ALERT_OFF")

mqtt_client.on_message = on_message

# -----------------------------------------------------------
# ASYNCIO MQTT LOOP
# -----------------------------------------------------------
async def mqtt_loop():
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)

    mqtt_client.loop_start()   # TIDAK blocking, AMAN DI STREAMLIT
    await asyncio.sleep(0.1)

# Jalankan 1x
asyncio.run(mqtt_loop())

# -----------------------------------------------------------
# UI START
# -----------------------------------------------------------
st.title("IoT ML Realtime Dashboard 🚀")

left, right = st.columns([1, 2])

# ===========================================================
# LEFT PANEL — STATUS + CONTROL
# ===========================================================
with left:
    st.subheader("Connection Status")
    st.metric("MQTT Connected", "Yes" if st.session_state.mqtt_connected else "No")
    st.metric("Broker", MQTT_BROKER)

    if st.session_state.last:
        st.subheader("Last Reading")
        st.write(pd.DataFrame([st.session_state.last]).T)
    else:
        st.info("Waiting for first sensor data...")

    st.subheader("Manual Override")
    col1, col2 = st.columns(2)
    if col1.button("Force ALERT ON"):
        st.session_state.override = "ON"
        mqtt_client.publish(TOPIC_OUTPUT, "ALERT_ON")
        st.success("Published ALERT_ON")

    if col2.button("Force ALERT OFF"):
        st.session_state.override = "OFF"
        mqtt_client.publish(TOPIC_OUTPUT, "ALERT_OFF")
        st.success("Published ALERT_OFF")

    if st.button("Clear override"):
        st.session_state.override = None
        st.info("Manual override cleared → Auto AI Alert ON")

    if st.button("Save Log CSV"):
        if st.session_state.logs:
            df = pd.DataFrame(st.session_state.logs)
            fn = "iot_log_" + datetime.utcnow().strftime("%Y%m%d%H%M%S") + ".csv"
            st.download_button("Download CSV", df.to_csv(index=False), file_name=fn)
        else:
            st.warning("No logs yet.")

# ===========================================================
# RIGHT PANEL — CHART + TABLE
# ===========================================================
with right:
    st.subheader("Live Sensor Chart")
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
    else:
        st.write("—")

# FOOTER
st.markdown("---")
st.write("Manual override:", st.session_state.override)
st.write("Total messages:", len(st.session_state.logs))
