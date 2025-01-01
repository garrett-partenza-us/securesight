import subprocess
import sys
import os

def start_client():
    """Start the Python client module."""
    print("Starting Python client...")
    client_process = subprocess.Popen([sys.executable, "-m", "client.run"])
    return client_process

def start_server():
    """Start the Go server."""
    print("Starting Go server...")
    # Change the directory to the 'server' folder, where the go.mod file is located
    os.chdir('server')  # This assumes 'server' is a subfolder of the current working directory
    # Run the Go server
    server_process = subprocess.Popen(["go", "run", "main.go", "ops.go"])
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

