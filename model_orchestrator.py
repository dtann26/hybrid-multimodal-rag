import subprocess
import time
import urllib.request
import urllib.error
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def wait_for_service(url: str, timeout: int = 600) -> bool:
    """Waits until the specified URL returns a 200 HTTP status."""
    start_time = time.time()
    logger.info(f"Waiting for service at {url} to become fully loaded...")
    while time.time() - start_time < timeout:
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req) as response:
                if response.getcode() == 200:
                    logger.info(f"Service at {url} is ready!")
                    return True
        except Exception:
            pass
        time.sleep(10)
    
    raise TimeoutError(f"Service at {url} did not become ready within {timeout} seconds.")

def start_individual_service(service_name: str):
    """Starts a specific service."""
    logger.info(f"Starting service: {service_name}...")
    subprocess.run(["docker", "compose", "up", "-d", service_name], check=True)
    logger.info(f"Service {service_name} started.")

def start_services():
    """Starts the vLLM services sequentially, waiting for each to spin up before moving to the next."""
    """Do not change the order of service. The embedding and reranker model can crash if its not the first to start."""
    logger.info("Starting base infrastructure (etcd, minio, standalone)...")
    start_individual_service("standalone")
    
    # Wait a few seconds for milvus to initialize
    time.sleep(15)

    logger.info("Starting Embedding model...")
    start_individual_service("vllm-embedding")
    wait_for_service("http://localhost:8002/v1/models")

    logger.info("Starting Reranker model...")
    start_individual_service("vllm-reranker")
    wait_for_service("http://localhost:8003/v1/models")

    logger.info("Starting LLM model...")
    start_individual_service("vllm-llm")
    wait_for_service("http://localhost:8001/v1/models")

    logger.info("All vLLM models loaded and running sequentially successfully!")

def stop_services():
    """Stops all services."""
    logger.info("Stopping all services...")
    subprocess.run(["docker", "compose", "down"], check=True)
    logger.info("All services stopped.")

def stop_individual_service(service_name: str):
    """Stops a specific service."""
    logger.info(f"Stopping service: {service_name}...")
    subprocess.run(["docker", "compose", "stop", service_name], check=True)
    logger.info(f"Service {service_name} stopped.")

if __name__ == "__main__":
    start_services()
