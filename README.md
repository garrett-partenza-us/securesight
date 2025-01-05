# Encrypted Facial Detection via Homomorphic Encryption in Golang

A client-server system for face prediction using **Fully Homomorphic Encryption (FHE)** with the **CKKS scheme** and **K-Nearest Neighbors (KNN)** for classifying faces. The client handles the video, face detection, feature extraction, and encryption, while the server calculates distances and predicts classes in an encrypted space. 🔒

![Video Sample](assets/demo.gif)

## Features

- **Client-Side (Go)**: Detect faces with **YOLO**, extract embeddings with **ResNet**, encrypt via **CKKS**, and send to the server.
- **Server-Side (Go)**: Compute Euclidean distance in the encrypted space using **CKKS** and return predictions to the client.
- **Concurrency**: Go routines for handling multiple faces and speeding things up.
- **Multi-face**: Handles multiple faces per frame.
- **Real-time**: Processes one frame every **1.7 seconds**.
- **Encryption FTW**: Sensitive data stays encrypted the whole time. 

## Prerequisites ⚡

- Go 1.18+
- [GoCV](https://github.com/hybridgroup/gocv)
- [Lattigo](https://github.com/tuneinsight/lattigo) for CKKS FHE
- ONNX runtime for YOLO and ResNet
- Python (for KNN model training)

## To Run
**Train Model**
```
cd model
python -m venv env
pip install -r requirements.txt
python train.py
```
**Client**
```
cd client
go run *.go
```
**Server**
```
cd server
go run *.go
```
