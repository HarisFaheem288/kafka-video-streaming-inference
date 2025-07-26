import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import base64
import io
import grpc
from concurrent import futures
import logging
import os
import time
from grpc_interface_pb2 import InferenceResponse
import grpc_interface_pb2_grpc
from flask import Flask, jsonify
import threading
import onnxruntime
from enum import Enum

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('inference_service.log')
    ]
)
logger = logging.getLogger(__name__)

# Optimization Types
class Optimization(Enum):
    TORCHSCRIPT = 1
    ONNX = 2
    VANILLA = 3

class ModelWrapper:
    """Optimized model wrapper with ONNX/TorchScript support"""
    def __init__(self, optimization=Optimization.ONNX):
        self.optimization = optimization
        self.model = None
        self.onnx_session = None
        self.transform = self._get_transforms()
        
        with open("imagenet_classes.txt") as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        self._initialize_model()

    def _get_transforms(self):
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _initialize_model(self):
        if self.optimization == Optimization.ONNX:
            self._init_onnx_model()
        elif self.optimization == Optimization.TORCHSCRIPT:
            self._init_torchscript_model()
        else:
            self._init_vanilla_model()

    def _init_vanilla_model(self):
        self.model = models.resnet18(pretrained=True)
        self.model.eval()
        logger.info("Initialized vanilla PyTorch model")

    def _init_torchscript_model(self):
        self.model = torch.jit.script(models.resnet18(pretrained=True))
        self.model.eval()
        logger.info("Initialized TorchScript optimized model")

    def _init_onnx_model(self):
        onnx_path = "resnet18.onnx"
        if not os.path.exists(onnx_path):
            self._export_onnx(onnx_path)
        
        self.onnx_session = onnxruntime.InferenceSession(
            onnx_path,
            providers=[
                'CUDAExecutionProvider', 
                'CPUExecutionProvider'
            ]
        )
        logger.info(f"Initialized ONNX model with providers: {self.onnx_session.get_providers()}")

    def _export_onnx(self, output_path):
        dummy_input = torch.randn(1, 3, 224, 224)
        model = models.resnet18(pretrained=True)
        model.eval()
        
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            opset_version=13,
            do_constant_folding=True
        )
        logger.info(f"Exported ONNX model to {output_path}")

    def predict(self, image_bytes):
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            input_tensor = self.transform(image).unsqueeze(0)
            
            start_time = time.perf_counter()
            
            if self.optimization == Optimization.ONNX:
                ort_inputs = {self.onnx_session.get_inputs()[0].name: input_tensor.numpy()}
                ort_outs = self.onnx_session.run(None, ort_inputs)
                predicted = torch.argmax(torch.tensor(ort_outs[0]), dim=1)
            else:
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    predicted = outputs.argmax(dim=1)
            
            latency = (time.perf_counter() - start_time) * 1000  # ms
            logger.info(f"Inference completed in {latency:.2f}ms")
            
            return self.classes[predicted.item()], latency
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}", exc_info=True)
            raise

# Flask App for Health/Metrics
health_app = Flask(__name__)
model_wrapper = ModelWrapper(optimization=Optimization.ONNX)

@health_app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "optimization": model_wrapper.optimization.name
    })

@health_app.route('/metrics')
def metrics():
    """Performance metrics endpoint"""
    return jsonify({
        "model": "resnet18",
        "optimization": model_wrapper.optimization.name,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    })

class InferenceService(grpc_interface_pb2_grpc.InferenceServicer):
    def Predict(self, request, context):
        try:
            img_data = base64.b64decode(request.image)
            label, latency = model_wrapper.predict(img_data)
            
            return InferenceResponse(
                label=label
            )
            
        except Exception as e:
            logger.error(f"gRPC prediction failed: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Prediction failed: {str(e)}")
            return InferenceResponse(label="ERROR", latency_ms=-1)

def serve():
    # Start Flask server in background
    flask_thread = threading.Thread(
        target=lambda: health_app.run(host='0.0.0.0', port=5000),
        daemon=True
    )
    flask_thread.start()
    
    # gRPC Server Configuration
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),
            ('grpc.max_send_message_length', 100 * 1024 * 1024)
        ]
    )
    
    grpc_interface_pb2_grpc.add_InferenceServicer_to_server(
        InferenceService(), server
    )
    server.add_insecure_port('[::]:50051')
    
    logger.info("Starting services:")
    logger.info(f"gRPC server listening on port 50051")
    logger.info(f"HTTP health checks at http://localhost:5000/health")
    logger.info(f"Model optimization: {model_wrapper.optimization.name}")
    
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()


# import torch
# import torchvision.transforms as transforms
# from torchvision import models
# from PIL import Image
# import base64
# import io
# import grpc
# from concurrent import futures
# from grpc_interface_pb2 import InferenceResponse
# import grpc_interface_pb2_grpc

# # Load pretrained model
# model = models.resnet18(pretrained=True)
# model.eval()

# with open("imagenet_classes.txt") as f:
#     classes = [line.strip() for line in f.readlines()]

# # Define transform
# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
# ])

# class InferenceService(grpc_interface_pb2_grpc.InferenceServicer):
#     def Predict(self, request, context):
    
#         # Decode base64 image
#         img_data = base64.b64decode(request.image)
#         image = Image.open(io.BytesIO(img_data)).convert("RGB")

#         # Preprocess
#         input_tensor = transform(image).unsqueeze(0)

#         # Predict
#         with torch.no_grad():
#             outputs = model(input_tensor)
#             _, predicted = outputs.max(1)
#             label = classes[predicted.item()]

#         return InferenceResponse(label=label)

# def serve():
#     server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
#     grpc_interface_pb2_grpc.add_InferenceServicer_to_server(InferenceService(), server)
#     server.add_insecure_port('[::]:50051')
#     server.start()
#     server.wait_for_termination()

# if __name__ == '__main__':
#     serve()
