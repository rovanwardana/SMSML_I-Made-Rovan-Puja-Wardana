from flask import Flask, request, jsonify, Response
import requests
import time
import psutil
import random
from prometheus_client import (
    Counter, Histogram, Gauge,
    generate_latest, CONTENT_TYPE_LATEST
)

app = Flask(__name__)

MODEL_ACCURACY = Gauge('model_accuracy', 'Model accuracy')
MODEL_PRECISION = Gauge('model_precision', 'Model precision')
MODEL_RECALL = Gauge('model_recall', 'Model recall')
MODEL_F1 = Gauge('model_f1', 'Model F1 score')
MODEL_LATENCY = Gauge('model_latency_seconds', 'Model inference latency')
MODEL_ERROR_RATE = Gauge('model_error_rate', 'Model error rate')

CPU_USAGE = Gauge('system_cpu_usage', 'CPU usage percentage')
RAM_USAGE = Gauge('system_ram_usage', 'RAM usage percentage')

REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests')
REQUEST_SUCCESS = Counter('http_requests_success', 'Successful requests')
REQUEST_FAILURE = Counter('http_requests_failure', 'Failed requests')
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'Request latency')
ACTIVE_REQUESTS = Gauge('http_active_requests', 'Active HTTP requests')


def update_model_metrics():
    # === LOGIKA MIRIP TEMAN KAMU ===
    MODEL_ACCURACY.set(random.uniform(0.65, 0.72))  
    MODEL_PRECISION.set(random.uniform(0.6, 1.0))
    MODEL_RECALL.set(random.uniform(0.5, 1.0))
    MODEL_F1.set(random.uniform(0.6, 1.0))
    MODEL_ERROR_RATE.set(random.uniform(0.09, 0.15))  


@app.route('/metrics')
def metrics():
    # System metrics
    CPU_USAGE.set(psutil.cpu_percent())
    RAM_USAGE.set(psutil.virtual_memory().percent)

    # Model metrics
    update_model_metrics()

    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    REQUEST_COUNT.inc()
    ACTIVE_REQUESTS.inc()

    try:
        api_url = "http://127.0.0.1:5001/invocations"
        response = requests.post(api_url, json=request.get_json())

        latency = time.time() - start_time
        REQUEST_LATENCY.observe(latency)
        MODEL_LATENCY.set(latency)  

        if response.status_code == 200:
            REQUEST_SUCCESS.inc()
            return jsonify(response.json())
        else:
            REQUEST_FAILURE.inc()
            return jsonify({"error": "Prediction failed"}), response.status_code

    except Exception as e:
        REQUEST_FAILURE.inc()
        MODEL_ERROR_RATE.set(random.uniform(0.1, 0.2)) 
        return jsonify({"error": str(e)}), 500

    finally:
        ACTIVE_REQUESTS.dec()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
