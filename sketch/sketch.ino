
// Sensor Acquisition + Motor Arbitration — MCU side
// Qwiic MPU-6050 on Wire1, defensive I2C reads
// Autonomous forward drive; safety arbitration overrides via alert level
// Test button + alert-tied buzzer for bench testing and audible feedback
#include <Wire.h>
#include <Arduino_RouterBridge.h>

// ---- HC-SR04 Pins ----
const int TRIG1 = 5;
const int ECHO1 = 3;
const int TRIG2 = 2;
const int ECHO2 = 11;

// ---- Sharp IR Pin ----
const int IR_PIN = A5;
const float IR_DIVIDER_RATIO = 0.667;

// ---- MPU-6050 ----
const int MPU_ADDR = 0x68;
const int PWR_MGMT_1 = 0x6B;
const int ACCEL_XOUT_H = 0x3B;
const float ACCEL_SCALE = 16384.0;
const float G_TO_MS2 = 9.81;

// ---- L293D Motor Pins ----
const int EN_LEFT  = 9;
const int IN1_LEFT = 8;
const int IN2_LEFT = 7;
const int EN_RIGHT = 10;
const int IN3_RIGHT = 12; // UNCONFIRMED: possibly needs swap w/ IN4_RIGHT
const int IN4_RIGHT = 13; // pending Pattern A/B diagnostic via test button

const int NORMAL_SPEED = 150;
const int SLOW_SPEED   = 80;

// ---- Loop Timing ----
const unsigned long LOOP_INTERVAL_MS = 20;
unsigned long lastLoopTime = 0;

float lastGoodAccelX = 0.0;

// ---- Arbitration State ----
int currentAlertLevel = 0;   // 0=CLEAR, 1=WARNING, 2=SLOW DOWN, 3=STOP
int manualCommand = 0;       // 0=stop, 1=forward, 2=backward, 3=left, 4=right

// ---- Test Toggle Button ----
const int TEST_BUTTON_PIN = 6;
bool lastButtonState = HIGH;
unsigned long lastDebounceTime = 0;
const unsigned long DEBOUNCE_MS = 50;
bool testDriveActive = false;
bool processedThisPress = false;

// ---- Buzzer ----
const int BUZZER_PIN = 4;
unsigned long lastBuzzerToggle = 0;
bool buzzerState = false;

void setup() {
  Serial.begin(9600);
  delay(2000);

  pinMode(EN_LEFT, OUTPUT);
  digitalWrite(EN_LEFT, LOW);
  pinMode(EN_RIGHT, OUTPUT);
  digitalWrite(EN_RIGHT, LOW);
  pinMode(IN1_LEFT, OUTPUT);
  pinMode(IN2_LEFT, OUTPUT);
  pinMode(IN3_RIGHT, OUTPUT);
  pinMode(IN4_RIGHT, OUTPUT);

  pinMode(TRIG1, OUTPUT);
  pinMode(ECHO1, INPUT);
  pinMode(TRIG2, OUTPUT);
  pinMode(ECHO2, INPUT);

  pinMode(TEST_BUTTON_PIN, INPUT_PULLUP);

  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);  // explicit LOW at boot — same boot-safety
                                   // philosophy already used for motor pins

  analogReadResolution(14);
  Wire1.begin();
  mpuWake();
  delay(500);

  Bridge.provide("set_alert", setAlertLevel);
  Bridge.provide("set_manual_command", setManualCommand);
}

void loop() {
  checkTestButton();
  updateBuzzer();

  unsigned long now = millis();
  if (now - lastLoopTime < LOOP_INTERVAL_MS) return;
  lastLoopTime = now;

  float dist1 = readUltrasonicCM(TRIG1, ECHO1);
  float dist2 = readUltrasonicCM(TRIG2, ECHO2);
  float irDist = readSharpIR();
  float accelX = readMPU6050_AccelX();

  sendPacket(dist1, dist2, irDist, accelX);
}


// Test Button
void checkTestButton() {
  bool reading = digitalRead(TEST_BUTTON_PIN);

  if (reading != lastButtonState) {
    lastDebounceTime = millis();
  }

  if ((millis() - lastDebounceTime) > DEBOUNCE_MS) {
    if (reading == LOW && !processedThisPress) {
      testDriveActive = !testDriveActive;
      manualCommand = testDriveActive ? 1 : 0;
      applyMotorOutput();
      processedThisPress = true;
      Serial.print("[TEST BUTTON] Drive toggled: ");
      Serial.println(testDriveActive ? "ON" : "OFF");
    } else if (reading == HIGH) {
      processedThisPress = false;
    }
  }

  lastButtonState = reading;
}


// Buzzer — alert-tied, non-blocking

void updateBuzzer() {
  unsigned long beepInterval;

  switch (currentAlertLevel) {
    case 0:
      digitalWrite(BUZZER_PIN, LOW);
      return;
    case 1:
      beepInterval = 500;
      break;
    case 2:
      beepInterval = 200;
      break;
    case 3:
      beepInterval = 80;
      break;
    default:
      digitalWrite(BUZZER_PIN, LOW);
      return;
  }

  if (millis() - lastBuzzerToggle >= beepInterval) {
    buzzerState = !buzzerState;
    digitalWrite(BUZZER_PIN, buzzerState ? HIGH : LOW);
    lastBuzzerToggle = millis();
  }
}


// Sensor Reads


float readUltrasonicCM(int trigPin, int echoPin) {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  long duration = pulseIn(echoPin, HIGH, 30000);
  if (duration == 0) return -1.0;
  return duration / 58.0;
}

float readSharpIR() {
  int raw = analogRead(IR_PIN);
  float dividedVoltage = (raw / 16383.0) * 3.3;
  float sensorVoltage = dividedVoltage / IR_DIVIDER_RATIO;
  if (sensorVoltage < 0.1) return -1.0;
  return 27.728 * pow(sensorVoltage, -1.2045);
}

void mpuWake() {
  Wire1.beginTransmission(MPU_ADDR);
  Wire1.write(PWR_MGMT_1);
  Wire1.write(0);
  Wire1.endTransmission(true);
}

float readMPU6050_AccelX() {
  Wire1.beginTransmission(MPU_ADDR);
  Wire1.write(ACCEL_XOUT_H);
  if (Wire1.endTransmission(false) != 0) return lastGoodAccelX;
  byte n = Wire1.requestFrom(MPU_ADDR, 2, true);
  if (n < 2) return lastGoodAccelX;
  int16_t rawX = (Wire1.read() << 8) | Wire1.read();
  lastGoodAccelX = (rawX / ACCEL_SCALE) * G_TO_MS2;
  return lastGoodAccelX;
}

void sendPacket(float us1, float us2, float ir, float ax) {
  Bridge.call("update_sensors", us1, us2, ir, ax);
}


// Motor Arbitration


void setAlertLevel(int level) {
  currentAlertLevel = level;
  applyMotorOutput();
}

void setManualCommand(int code) {
  manualCommand = code;
  applyMotorOutput();
}

void applyMotorOutput() {
  int speed = NORMAL_SPEED;

  if (currentAlertLevel == 3 && manualCommand == 1) {
    stopAll();
    return;
  }
  if (currentAlertLevel == 2 && manualCommand == 1) {
    speed = SLOW_SPEED;
  }

  switch (manualCommand) {
    case 1:
      digitalWrite(IN1_LEFT, HIGH); digitalWrite(IN2_LEFT, LOW);
      digitalWrite(IN3_RIGHT, HIGH); digitalWrite(IN4_RIGHT, LOW);
      analogWrite(EN_LEFT, speed); analogWrite(EN_RIGHT, speed);
      break;
    case 2:
      digitalWrite(IN1_LEFT, LOW); digitalWrite(IN2_LEFT, HIGH);
      digitalWrite(IN3_RIGHT, LOW); digitalWrite(IN4_RIGHT, HIGH);
      analogWrite(EN_LEFT, NORMAL_SPEED); analogWrite(EN_RIGHT, NORMAL_SPEED);
      break;
    case 3:
      digitalWrite(IN1_LEFT, LOW); digitalWrite(IN2_LEFT, HIGH);
      digitalWrite(IN3_RIGHT, HIGH); digitalWrite(IN4_RIGHT, LOW);
      analogWrite(EN_LEFT, NORMAL_SPEED); analogWrite(EN_RIGHT, NORMAL_SPEED);
      break;
    case 4:
      digitalWrite(IN1_LEFT, HIGH); digitalWrite(IN2_LEFT, LOW);
      digitalWrite(IN3_RIGHT, LOW); digitalWrite(IN4_RIGHT, HIGH);
      analogWrite(EN_LEFT, NORMAL_SPEED); analogWrite(EN_RIGHT, NORMAL_SPEED);
      break;
    default:
      stopAll();
  }
}

void stopAll() {
  analogWrite(EN_LEFT, 0);
  analogWrite(EN_RIGHT, 0);
}