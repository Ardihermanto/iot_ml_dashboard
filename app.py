import streamlit as st
import asyncio
import json
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import plotly.graph_objs as go
from asyncio_mqtt import Client, MqttError

# UI CONFIG
st.set_page_config(page_title="IoT ML Dashboard", layout="wide")

# --------------------------
# CONFIG
# --------------------------
MQTT_BROKER = st.secrets.get("MQTT_BROKER", "broker.emqx.io")
MQTT_PORT   = int(st.secrets.get("MQTT_PORT", 8084))
TOPIC_SENSOR = st.secrets.get("TOPIC_SENSOR", "iot/class/session5/sensor")
TOPIC_OUTPUT = st.secrets.get("TOPIC_OUTPUT", "iot/class/session5/output")
MODEL_PATH = st.secrets.get("MODEL_PATH", "iot_temp_model.pkl")

# --------------------------
# LOAD MODEL
# --------------------------
@st.cache_resource
def load_model(path):
    return joblib.load(path)

model = load_model(MODEL_PATH)

# --------------------------
# SESSION STATE
# --------------------------
if "logs" not in st.session_state:
    st.session_state.logs = []

if "last" not in st.session_state:
    st.session_state.last = None

if "mqtt_connected" not in st.session_state:
    st.session_state.mqtt_connected = False

# --------------------------
# ASYNC MQTT LOOP (NO THREAD)
# --------------------------
async def mqtt_listener():
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

                    # ML inference
                    X = [[temp, hum]]
                    pred = model.predict(X)[0]

                    try:
                        conf = float(np.max(model.predict_proba(X)))
                    except:
                        conf = None

                    row = {
                        "ts": ts,
                        "temp": temp,
                        "hum": hum,
                        "pred": pred,
                        "conf": conf
                    }

                    st.session_state.logs.append(row)
                    st.session_state.last = row

                    # publish output
                    if pred == "Panas":
                        await client.publish(TOPIC_OUTPUT, "ALERT_ON")
                    else:
                        await client.publish(TOPIC_OUTPUT, "ALERT_OFF")

    except MqttError as e:
        st.session_state.mqtt_connected = False
        await asyncio.sleep(3)
        await mqtt_listener()  # reconnect


# --------------------------
# START ASYNC LOOP (SAFE)
# --------------------------
asyncio.run(mqtt_listener())


# --------------------------
# UI LAYOUT
# --------------------------
st.title("IoT ML Realtime Dashboard")

col1, col2 = st.columns([1,2])

with col1:
    st.subheader("Connection Status")
    st.metric("MQTT Broker", MQTT_BROKER)
    st.metric("Connected", "Yes" if st.session_state.mqtt_connected else "No")

    st.subheader("Last Reading")
    if st.session_state.last:
        st.write(pd.DataFrame([st.session_state.last]).T)
    else:
        st.info("Waiting for sensor data...")

    if st.button("Save Log to CSV"):
        df = pd.DataFrame(st.session_state.logs)
        st.download_button("Download CSV",
                           df.to_csv(index=False).encode(),
                           file_name="iot_log.csv")

with col2:
    st.subheader("Live Chart")

    df = pd.DataFrame(st.session_state.logs)
    if not df.empty:
        df_plot = df.tail(200)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_plot["ts"], y=df_plot["temp"], name="Temp"))
        fig.add_trace(go.Scatter(x=df_plot["ts"], y=df_plot["hum"], name="Humidity"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data yet")

st.markdown("---")
st.write("Total messages:", len(st.session_state.logs))
