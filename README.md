# STM32F407 IoT Weather Station with TinyML Trend Predictor

This project outlines how to build an edge-AI weather station using an **STM32F407** microcontroller, **DHT11** (temperature & humidity), and **BMP280** (barometric pressure) sensors. It utilizes **TinyML** (TensorFlow Lite for Microcontrollers / STM32Cube.AI) to predict short-term weather trends (e.g., *Clear, Cloudy, Rain Incoming, Storm*) directly on the microcontroller without needing cloud inference.

---

## 📋 Table of Contents
1. [System Architecture](#-system-architecture)
2. [Hardware Requirements & Pinout](#-hardware-requirements--pinout)
3. [Phase 1: Data Collection & Logging](#-phase-1-data-collection--logging)
4. [Phase 2: Model Training & Quantization (Google Colab)](#-phase-2-model-training--quantization-google-colab)
5. [Phase 3: STM32Cube.AI Integration & Firmware](#-phase-3-stm32cubeai-integration--firmware)
6. [Phase 4: IoT Connectivity (Optional)](#-phase-4-iot-connectivity-optional)

---

## 🏗 System Architecture

1. **Sensing:** STM32F407 reads Temperature, Humidity (DHT11), and Pressure (BMP280) at regular intervals (e.g., every 15 minutes).
2. **Feature Engineering:** The MCU maintains a sliding window buffer (e.g., the last 3 hours of readings) and calculates rates of change ($\Delta P / \Delta t$, $\Delta T / \Delta t$).
3. **Edge Inference:** The normalized buffer is fed into a quantized `.tflite` neural network running on the ARM Cortex-M4 FPU.
4. **Output:** The model outputs probabilities for weather trends, which can be displayed locally or transmitted via an IoT module (ESP8266/ESP32) to an MQTT broker.

---

## 🔌 Hardware Requirements & Pinout

* **Microcontroller:** STM32F407VG (e.g., STM32F407G-DISC1 or Black Pill F407)
* **Sensors:**
  * **BMP280:** Barometric Pressure & Temperature (I2C)
  * **DHT11:** Relative Humidity & Temperature (Single-Wire Digital)
* **IoT Module (Optional):** ESP-01 (ESP8266) or SIM800L via UART

### Recommended Wiring

| Component | Pin | STM32F407 Pin | Interface / Notes |
| :--- | :--- | :--- | :--- |
| **BMP280** | VCC | 3.3V | Power |
| | GND | GND | Ground |
| | SCL | `PB6` | I2C1_SCL |
| | SDA | `PB7` | I2C1_SDA |
| **DHT11** | VCC | 3.3V | Power |
| | GND | GND | Ground |
| | DATA | `PA1` | GPIO Output/Input (Requires 4.7k$\Omega$ pull-up) |
| **USB-TTL / Debug** | TX/RX | `PA2` / `PA3` | USART2 (For logging data to PC) |

---

## 📊 Phase 1: Data Collection & Logging

Before training a model, you need historical data correlating sensor readings with actual weather outcomes.

### Option A: Use Public Historical Weather Data (Fastest)
Download hourly weather datasets (Temperature, Humidity, Pressure, Weather Description) from sources like **Kaggle**, **Open-Meteo**, or **NOAA** for your specific geographic region. Ensure the units match your sensors ($^\circ\text{C}$, $\%$, and $\text{hPa}$).

### Option B: Log Real Data from Your STM32
1. Flash a basic firmware that reads the DHT11 and BMP280 every 10–15 minutes.
2. Print the values over UART in CSV format:
   ```text
   timestamp,temp_c,humidity_pct,pressure_hpa
   1700000000,28.4,65.0,1012.35
   ```
3. Use a Python script (`pyserial`) on your PC or an SD card module to log this data over several weeks, manually or automatically tagging the weather conditions.

---

## 🧠 Phase 2: Model Training & Quantization (Google Colab)

You train the model in Python using TensorFlow, quantize it to fit into the STM32's Flash/RAM, and convert it into a C byte array.

### 1. Feature Engineering
Weather prediction relies heavily on **trends** rather than single instantaneous readings. Structure your input features as a time-series window (e.g., $N = 6$ readings taken 30 minutes apart = 3 hours of history):
* Input Shape: `(6, 3)` $\rightarrow$ 6 time steps $\times$ 3 features (`[Temperature, Humidity, Pressure]`).
* **Crucial:** Normalize your inputs (e.g., Min-Max scaling between `0.0` and `1.0` or Z-score standardization). **Save the exact scaling parameters (min/max or mean/std)**—you will need to hardcode them into your STM32 C code!

### 2. Build & Train the Model (Colab Snippet)
A lightweight 1D-CNN or Multi-Layer Perceptron (MLP) works best for tabular time-series on Cortex-M4:

```python
import tensorflow as tf
import numpy as np

# Example Model Architecture (Input: 6 timesteps x 3 sensors = 18 flattened features)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(18,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax') # 4 Classes: Clear, Cloudy, Rain, Storm
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))
```

### 3. Convert & Quantize to TFLite
The STM32F407 has 1 MB of Flash and 192 KB of RAM, which is plenty for TinyML, though `int8` quantization makes inference significantly faster.

```python
# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optional: Enable Dynamic Range or Full Integer Quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the .tflite file
with open('weather_model.tflite', 'wb') as f:
    f.write(tflite_model)
```

### 4. Convert `.tflite` to a C Byte Array
If you are using **TensorFlow Lite for Microcontrollers (TFLM)** directly, convert the binary file into a C header using `xxd` in Colab:

```bash
!xxd -i weather_model.tflite > weather_model_data.h
```
*(Note: If you use **STM32Cube.AI** in Phase 3, you can simply upload the `weather_model.tflite` file directly into the GUI, and it will generate optimized C code automatically!)*

---

## ⚙️ Phase 3: STM32Cube.AI Integration & Firmware

The easiest and most optimized way to deploy your model on an STM32F407 is using STMicroelectronics' **X-CUBE-AI** package inside **STM32CubeIDE**.

### Step 1: Configure Hardware in STM32CubeMX
1. Open **STM32CubeIDE** and create a new project for `STM32F407VGTx`.
2. Enable **I2C1** (`PB6`, `PB7`) for the BMP280.
3. Configure `PA1` as a GPIO Output/Input and configure a hardware Timer (e.g., **TIM2** with 1 $\mu\text{s}$ tick) to handle the microsecond delays required by the DHT11 protocol.
4. Enable **USART2** (`PA2`, `PA3`) for serial debugging.
5. Enable **CRC** (Cyclic Redundancy Check) under *Computing* — **X-CUBE-AI requires the CRC peripheral to be enabled.**

### Step 2: Import the TinyML Model
1. In CubeMX, click **Software Packs** $\rightarrow$ **Select Components**.
2. Install and enable **STMicroelectronics.X-CUBE-AI** (Core + Application Template).
3. Go to the **X-CUBE-AI** configuration tab:
   * Click **Add Network**.
   * Select **TFLite** as the model format and browse to your `weather_model.tflite` file.
   * Click **Analyze** to verify RAM/Flash usage and ensure all layers are supported.
4. Click **Generate Code** (Alt+K).

### Step 3: Firmware Inference Workflow
In your `main.c` application loop, implement the following pipeline:

```c
#include "ai_platform.h"
#include "network.h"
#include "network_data.h"

// Circular buffer holding the last 6 readings (Temp, Hum, Press)
float sensor_window[6][3]; 

// Scaling constants from your Python training script
const float MIN_VALS[3] = { -10.0f, 0.0f, 950.0f };
const float MAX_VALS[3] = {  50.0f, 100.0f, 1050.0f };

void Run_Weather_Inference(void) {
    float input_data[18];
    float output_probs[4];

    // 1. Flatten and Normalize the sliding window
    int idx = 0;
    for (int t = 0; t < 6; t++) {
        for (int f = 0; f < 3; f++) {
            float val = sensor_window[t][f];
            input_data[idx++] = (val - MIN_VALS[f]) / (MAX_VALS[f] - MIN_VALS[f]);
        }
    }

    // 2. Run X-CUBE-AI Inference (wrapper generated by CubeMX)
    MX_X_CUBE_AI_Process(input_data, output_probs);

    // 3. Interpret Results
    // output_probs[0] = Clear, [1] = Cloudy, [2] = Rain, [3] = Storm
    printf("Forecast -> Clear: %.1f%% | Cloudy: %.1f%% | Rain: %.1f%% | Storm: %.1f%%\r\n",
           output_probs[0]*100, output_probs[1]*100, output_probs[2]*100, output_probs[3]*100);
}
```

---

## 🌐 Phase 4: IoT Connectivity (Optional)

Since the STM32F407 does not have built-in Wi-Fi, you can connect an **ESP8266 (ESP-01)** or **ESP32** to `USART3` (`PB10`/`PB11`):
1. Send AT commands (`AT+CIPSTART`, `AT+CIPSEND`) or a lightweight JSON payload over UART to the ESP module.
2. Have the ESP module publish the current sensor readings alongside the TinyML weather prediction to an **MQTT Broker** (e.g., Mosquitto, HiveMQ) or **Home Assistant**.

---

## 💡 Tips for Better Accuracy
* **Upgrade DHT11 to DHT22 or BME280:** The DHT11 has $\pm 2^\circ\text{C}$ and $\pm 5\%$ humidity accuracy, which introduces noise into ML models. Since you already have a BMP280, use the **BMP280's temperature sensor** (which is much more accurate) instead of the DHT11's temperature reading!
* **Barometric Pressure Delta:** Adding a calculated feature like `Pressure_Now - Pressure_3Hours_Ago` significantly boosts the neural network's ability to detect incoming storms.
