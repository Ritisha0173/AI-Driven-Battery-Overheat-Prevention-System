#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// TinyML
#include "model_data.cc"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// OLED
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

// Pins
#define CURRENT_PIN 34
#define VOLTAGE_PIN 35
#define LED_PIN 2
#define BUZZER_PIN 15

// Battery constants (adjust if needed)
float R_internal = 0.05;     // ohm
float thermal_res = 10.0;   // °C/W
float ambient_temp = 28.0;  // °C

// TinyML
constexpr int kTensorArenaSize = 20 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;

// ---------------- SENSOR FUNCTIONS ----------------
float readCurrent() {
  int adc = analogRead(CURRENT_PIN);
  float voltage = (adc / 4095.0) * 3.3;
  float current = (voltage - 2.5) / 0.066; // ACS712 example
  return abs(current);
}

float readVoltage() {
  int adc = analogRead(VOLTAGE_PIN);
  float voltage = (adc / 4095.0) * 3.3;
  return voltage * 5; // voltage divider
}

// Estimate temperature
float calculateTemperature(float current) {
  float power_loss = current * current * R_internal;
  return ambient_temp + (power_loss * thermal_res);
}

// ---------------- SETUP ----------------
void setup() {
  Serial.begin(115200);

  pinMode(LED_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);

  // OLED init
  display.begin(SSD1306_SWITCHCAPVCC, 0x3C);
  display.clearDisplay();
  display.setTextColor(WHITE);
  display.setTextSize(2);

  // TinyML init
  const tflite::Model* model = tflite::GetModel(battery_temp_model_tflite);
  static tflite::AllOpsResolver resolver;

  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize);

  interpreter = &static_interpreter;
  interpreter->AllocateTensors();

  input = interpreter->input(0);
  output = interpreter->output(0);
}

// ---------------- LOOP ----------------
void loop() {
  float current = readCurrent();
  float voltage = readVoltage();
  float temperature = calculateTemperature(current);

  // ML input
  input->data.f[0] = current;
  input->data.f[1] = voltage;
  input->data.f[2] = temperature;

  interpreter->Invoke();
  float prediction = output->data.f[0];

  // Alert logic
  if (prediction > 0.5) {   // ML predicts temp > 32°C
    digitalWrite(LED_PIN, HIGH);
    digitalWrite(BUZZER_PIN, HIGH);
  } else {
    digitalWrite(LED_PIN, LOW);
    digitalWrite(BUZZER_PIN, LOW);
  }

  // OLED display (ONLY temperature)
  display.clearDisplay();
  display.setCursor(0, 25);
  display.print("Temp:");
  display.print(temperature, 1);
  display.print("C");
  display.display();

  delay(1000);
}

