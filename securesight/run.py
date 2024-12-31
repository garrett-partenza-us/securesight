import subprocess
import sys

def start_client():
    """Start the Python client module."""
    print("Starting Python client...")
    client_process = subprocess.Popen([sys.executable, "-m", "client.run"])
    return client_process

def start_server():
    """Start the Go server."""
    print("Starting Go server...")
    server_process = subprocess.Popen(["go", "run", "server/main.go server/ops.go"])
    return server_process

def main():
    # Start both the client and the server
    client_process = start_client()
    server_process = start_server()

    # Wait for both processes to complete
    try:
        client_process.wait()
        server_process.wait()
    except KeyboardInterrupt:
        print("Terminating processes...")
        client_process.terminate()
        server_process.terminate()
        client_process.wait()
        server_process.wait()

if __name__ == "__main__":
    main()
