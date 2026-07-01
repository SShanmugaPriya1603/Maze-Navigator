#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <LiquidCrystal_I2C.h>

LiquidCrystal_I2C lcd(0x27, 16, 2);
Adafruit_MPU6050 mpu;

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);

  lcd.init();
  lcd.backlight();
  lcd.setCursor(0, 0);
  lcd.print("Gesture Ready!");

  if (!mpu.begin()) {
    Serial.println("MPU6050 not found!");
    while (1);
  }
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  Serial.println("Ready!");
}

void loop() {
  if (Serial.available()) {
    String gesture = Serial.readStringUntil('\n');
    gesture.trim();
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("Gesture:");
    lcd.setCursor(0, 1);
    lcd.print(gesture);
  }

  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);
  Serial.print(a.acceleration.x); Serial.print(",");
  Serial.print(a.acceleration.y); Serial.print(",");
  Serial.println(a.acceleration.z);

  delay(20);
}