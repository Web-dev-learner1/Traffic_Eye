#  TrafficEye: Intelligent Traffic Monitoring System  

##  Overview  
TrafficEye is a real-time traffic monitoring system that detects and tracks vehicles using OpenCV and dynamically controls a simulated traffic light based on vehicle density. This project aims to improve traffic management using computer vision techniques.  

##  Features  
-  **Vehicle Detection & Tracking** – Uses background subtraction and contour detection to identify moving objects.  
-  **Traffic Density Monitoring** – Tracks the number of vehicles in a region of interest (ROI).  
-  **Smart Traffic Light Control** – Changes the traffic signal dynamically based on vehicle count.  
-  **Real-Time Processing** – Efficiently processes video streams for accurate traffic monitoring.  

## 📂 Project Structure  
├── tracker.py # Euclidean Distance Tracker for object tracking

├── traffic_monitor.py # Main script for vehicle detection & traffic control

├── traffic_1.mp4 # Sample video for testing

└── README.md # Project documentation

##  Technologies Used  
- **Programming Language:** Python  
- **Libraries:** OpenCV (cv2), NumPy  
- **Algorithm:** Background Subtraction (MOG2), Contour Detection, Object Tracking  
