import streamlit as st
import base64
import grpc
import numpy as np
import cv2
import time
from kafka import KafkaConsumer
import grpc_interface_pb2
import grpc_interface_pb2_grpc
# import grpc_interface_pb2
# import grpc_interface_pb2_grpc
import threading
from queue import Queue

# Streamlit settings
st.set_page_config(page_title="Real-Time Inference", layout="wide")
st.title("🎥 Real-Time Webcam Stream Inference")

# UI placeholders
frame_display = st.empty()
prediction_display = st.empty()
latency_display = st.empty()
fps_display = st.empty()
frame_counter = st.empty()
stop_button = st.button("🛑 Stop Inference Early")

# Shared variables
latest_frame = None
latest_prediction = ""
latest_latency = 0.0
frame_count = 0
total_latency = 0.0
start_time = time.time()
should_stop = False

# Create a queue for frames
frame_queue = Queue()

def kafka_consumer_thread():
    global latest_frame, latest_prediction, latest_latency, frame_count, total_latency, should_stop
    
    consumer = KafkaConsumer(
        'webcam-frames',
        bootstrap_servers='kafka:9092',  # Use service name
        auto_offset_reset='latest',
        group_id='streamlit-consumer'
    )

  # Use Docker service name
    channel = grpc.insecure_channel('inference-service:50051')
    stub = grpc_interface_pb2_grpc.InferenceStub(channel)

    for msg in consumer:
        if should_stop:
            break

        # Decode frame
        img_bytes = base64.b64decode(msg.value)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Prepare for gRPC
        _, jpeg_bytes = cv2.imencode('.jpg', frame)
        jpeg_base64 = base64.b64encode(jpeg_bytes).decode('utf-8')
        request = grpc_interface_pb2.InferenceRequest(image=jpeg_base64)

        # gRPC call
        t1 = time.time()
        response = stub.Predict(request)
        latency = time.time() - t1

        # Update shared variables
        latest_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        latest_prediction = response.label
        latest_latency = latency
        frame_count += 1
        total_latency += latency

# Start Kafka consumer in a separate thread
thread = threading.Thread(target=kafka_consumer_thread, daemon=True)
thread.start()

# Main display loop
while True:
    if stop_button:
        should_stop = True
        st.warning("🛑 Inference stopped manually.")
        break

    elapsed = time.time() - start_time
    if elapsed > 120:  # 30 seconds
        st.success("✅ 120 seconds of inference completed.")
        should_stop = True
        break

    if latest_frame is not None:
        # Display in Streamlit
        frame_display.image(latest_frame, channels="RGB", caption="📷 Live Frame")
        prediction_display.success(f"Prediction: {latest_prediction}")
        latency_display.info(f"⏱️ Latency: {latest_latency:.3f} seconds")
        
        # Calculate FPS and average latency
        fps = frame_count / (time.time() - start_time)
        avg_latency = total_latency / frame_count if frame_count > 0 else 0
        
        fps_display.warning(f"🚀 FPS: {fps:.2f}")
        frame_counter.markdown(f"🖼️ Total frames: **{frame_count}**")

    time.sleep(0.01)  # Small delay to prevent high CPU usage

# Summary
st.markdown("---")
st.subheader("📊 Inference Summary")
st.write(f"🖼️ Total frames: {frame_count}")
# st.write(f"🚀 Throughput (FPS): {fps:.2f}")