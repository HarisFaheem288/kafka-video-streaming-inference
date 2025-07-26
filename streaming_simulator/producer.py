import cv2
import base64
from kafka import KafkaProducer
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_kafka_producer():
    while True:
        try:
            return KafkaProducer(
                bootstrap_servers='kafka:9092',
                max_block_ms=5000  # Wait up to 5 seconds
            )
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}. Retrying...")
            time.sleep(2)

def produce_frames():
    producer = create_kafka_producer()
    cap = cv2.VideoCapture("video.mp4")
    
    if not cap.isOpened():
        logger.error("Error opening video file")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                # Reset video to beginning when it ends
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Process frame
            _, buffer = cv2.imencode('.jpg', frame)
            jpg_as_text = base64.b64encode(buffer)
            
            try:
                producer.send('webcam-frames', jpg_as_text)
                logger.info("Frame sent to Kafka")
            except Exception as e:
                logger.error(f"Error sending to Kafka: {e}")
                # Recreate producer if connection fails
                producer = create_kafka_producer()
                continue
            
            # Control frame rate (30 FPS)
            time.sleep(1/30)
            
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        producer.close()
        logger.info("Producer stopped")

if __name__ == "__main__":
    while True:
        logger.info("Starting frame producer")
        produce_frames()

        time.sleep(1)