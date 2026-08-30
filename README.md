# Sensor Fusion Collision Avoidance System

Real-time multi-sensor fusion for autonomous collision avoidance, built on a dual-processor embedded platform. Built for ECE4520 Automotive Mechatronics I, Oakland University.

**Team:** Sergio Gamez, Narsai Ibrahim, Connor Sitarski

---

## What This Is

A small 4WD RC vehicle that fuses four independent sensors: two ultrasonic rangefinders, an infrared distance sensor, an IMU, and a camera running object detection into a single Kalman-filtered estimate of obstacle distance and closing velocity. That estimate drives a time-to-collision (TTC) calculation, which governs a tiered, safety-first motor response: normal speed, reduced speed, or full stop.

No single sensor is trusted unconditionally. Each one is weighted by an empirically-assigned trust factor in the Kalman filter, the same fusion philosophy used in real automotive ADAS systems, implemented here on real embedded hardware with real electrical and timing constraints.

## Architecture

The system runs across the Arduino UNO Q's two onboard processors, connected by its built-in Bridge RPC:

```
┌─────────────────────────┐         ┌──────────────────────────────┐
│   MCU (Zephyr/STM32U585) │         │   Linux Side (Debian/QRB2210) │
│                          │         │                                │
│  Sensor polling (50Hz)   │──Bridge→│  Kalman filter fusion          │
│  Motor arbitration       │←Bridge──│  TTC / alert-level computation │
│  Safety override logic   │         │  Camera capture + YOLOv5n CV   │
│  Test button / buzzer    │         │  Flask live dashboard           │
└─────────────────────────┘         └──────────────────────────────┘
```

The MCU owns the hard real-time path — polling all sensors every 20ms and applying motor output immediately based on the latest safety alert. The Linux side owns everything computationally heavier: sensor fusion math, the camera/CV pipeline, and a live telemetry dashboard.

## Hardware

| Component | Role |
|---|---|
| Arduino UNO Q 4GB | Dual-processor controller (STM32U585 + Qualcomm QRB2210) |
| HC-SR04 Ultrasonic ×2 | Primary distance measurement, front-left and front-right |
| Sharp GP2Y0A21 IR | Secondary short-range distance measurement |
| MPU-6050 IMU | Vehicle acceleration input (I2C / Qwiic) |
| Logitech C270 Webcam | Camera input for object-detection distance estimate |
| L293D H-Bridge ×2 | Motor driver |
| 7.4V LiPo Battery | Power source, split across raw/regulated rails |

Full signal-conditioning and power-distribution details are in [`docs/`](./docs).

## Software

- **Kalman filter**: two-state filter tracking `[distance, velocity]`, with per-sensor trust weights (`R = 4.0` ultrasonic, `9.0` IR, `25.0` camera)
- **Computer vision**: YOLOv5n exported to ONNX, run through OpenCV's DNN module across three independent threads (camera grab, inference, main loop) so inference latency never blocks the real-time sensor path
- **Motor arbitration**: MCU-side safety logic: a STOP alert overrides the drive command and halts the vehicle; a SLOW DOWN alert caps speed without blocking it
- **Live dashboard**: Flask-served MJPEG video feed with detection overlays, plus real-time distance/speed/TTC telemetry

## Results
| Metric | Value |
|---|---|
| Camera inference speedup | 3× (190ms → ~60ms/frame) after explicit OpenCV backend/target pinning |
| Safety loop rate | 50Hz, never blocked by the CV pipeline |
| Sensors fused | 4 (ultrasonic ×2, IR, camera) |

Full results, methodology, and known limitations are documented in [`docs/report.md`](./docs/report.md).

## Repository Structure

```
├── sketch/
│   └── sketch.ino          # MCU firmware — sensor polling, motor arbitration
├── python/
│   ├── main.py              # Linux-side fusion, CV pipeline, dashboard
│   ├── requirements.txt
├── app.yaml                 # App Lab container config (ports, name)

```

## Known Limitations

- The camera's distance-from-bounding-box formula uses a placeholder calibration constant, not yet empirically calibrated across multiple measured distances
- A Bluetooth manual-drive mode was designed but not completed; the deployed container environment doesn't support the `AF_BLUETOOTH` socket family required for classic RFCOMM communication
- GPU/NPU-accelerated inference via the board's onboard Adreno GPU was investigated but not pursued, pending confirmation that the required vendor SDK is available on this board's OS image




