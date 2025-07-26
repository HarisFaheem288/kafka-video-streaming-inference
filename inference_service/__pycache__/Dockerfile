# Stage 1: Build environment
FROM python:3.9-slim as builder

WORKDIR /app
COPY requirement_inference.txt .


RUN pip install --user -r requirement_inference.txt && \
    rm -rf /root/.cache/pip

# Stage 2: Runtime environment
FROM python:3.9-slim

WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY inference_service/ .

RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. grpc_interface.proto

EXPOSE 50051
EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl -f http://localhost:5000/health || exit 1

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

CMD ["python", "server.py"]
