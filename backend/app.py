import os
from flask import Flask, jsonify
import redis

app = Flask(__name__)

redis_host = os.environ.get("REDIS_HOST", "localhost")
redis_port = int(os.environ.get("REDIS_PORT", 6379))
redis_password = os.environ.get("REDIS_PASSWORD", "")

redis_client = redis.Redis(
    host=redis_host,
    port=redis_port,
    password=redis_password if redis_password else None,
    decode_responses=True,
)


@app.route("/api/ping")
def ping():
    return jsonify({"status": "ok"})


@app.route("/api/health")
def health():
    try:
        redis_client.ping()
        redis_status = "connected"
    except Exception:
        redis_status = "disconnected"
    return jsonify({"redis": redis_status})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
