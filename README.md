# 🛡️ Sentinel: Autonomous Workplace Safety Monitor
*(Hackathon Submission for HNX26PSI07: Autonomous Vision & Behaviour Understanding)*

## 1. What problem does this system solve?
Finding people in a video feed is easy; understanding if they are safe is hard. 
Currently, large factories and warehouses rely on human security guards staring at walls of CCTV monitors. Human attention degrades rapidly after just 20 minutes, meaning critical safety events—like a worker collapsing from cardiac arrest in a quiet aisle, or an employee running dangerously near heavy machinery—often go unnoticed until it's too late. **Sentinel** solves this by turning passive CCTV cameras into active, tireless safety monitors that instantly flag dangerous behaviors.

## 2. Who is the target & end customer?
* **Target Industry:** Manufacturing, Warehousing, Logistics, and Heavy Construction.
* **The Buyer (Customer):** Health, Safety, and Environment (HSE) Directors or Facility Operations Managers who are financially liable for workplace accidents and injuries.
* **The End User:** The on-site Security/Safety Operations Center (SOC) personnel who monitor the dashboard.

## 3. How is the system going to be? (User Experience)
The system acts as a "second set of eyes." It does not replace the security team; it augments them. 
Instead of staring at 50 screens, the user stares at a clean dashboard. Most of the time, the dashboard is quiet. When the AI detects a safety violation (e.g., a worker running, or someone completely inactive for 5 minutes), the dashboard flashes a real-time alert (e.g., `[14:02:11] CRITICAL: Person #4 Inactive in Zone B - Possible Injury`). 

## 4. What is the user input?
* **Current Prototype Input:** The user uploads a raw `.mp4` surveillance video file into our Streamlit web application.
* **Production Input (Future):** The system connects directly to the facility's existing RTSP IP camera streams (No new hardware required!). The user just inputs the IP camera URL into the dashboard configuration.

## 5. Roadmap
* **Phase 1 (Hackathon MVP):** Detect Running (Speed Violations) and Inactivity (Falls/Injuries) using heuristic spatial tracking.
* **Phase 2 (Zone Mapping):** Allow users to draw polygons on the camera feed to designate "Restricted Zones" (e.g., around a forklift path) and trigger alerts if a person enters them.
* **Phase 3 (Pose Estimation):** Upgrade the YOLO model to YOLO-Pose to detect the anatomical structure of the workers. This will allow us to detect specific ergonomic violations (e.g., lifting heavy boxes with the back instead of the knees) and lack of PPE (e.g., no hardhat).

---

## 💻 Hackathon Setup & Reproduction Guide

### Technologies Used
* **Core Vision Model:** YOLOv8 (Ultralytics) for zero-shot person detection.
* **Tracker:** ByteTrack for persistent ID assignment across frames.
* **Logic Engine:** Python & OpenCV (Custom Heuristic Math for Speed & Inactivity).
* **Frontend:** Streamlit.

### How to Install & Run
1. Clone this repository.
2. Ensure you have Python 3.9+ installed.
3. Install the dependencies:
   ```bash
   pip install ultralytics opencv-python-headless numpy streamlit
   ```
4. Run the interactive web dashboard:
   ```bash
   streamlit run app.py
   ```
5. Upload the provided `sample.mp4` to the dashboard to see the real-time event logging and bounding boxes in action.

### The "Core Reasoning" (How we avoided hallucination)
We explicitly avoided training a "black box" neural network that just spits out an anomaly score. Instead, we used a **Modular Heuristic Pipeline**. We use Deep Learning (YOLO) exclusively to find the $X, Y$ coordinates of humans. We then use strict, verifiable Euclidean math to calculate velocity (pixels per frame) and clustering (inactivity radius) to flag behaviors. This guarantees 100% explainability for every single alert generated.
