/*
 * ESP32 motor controller firmware for the LGControl project.
 *
 * Implements exactly the serial protocol that Motor/src/motor_interface.py
 * (ESP32MotorInterface) expects:
 *
 *   PC  -> ESP32: PING          ESP -> PC: PONG
 *   PC  -> ESP32: SET_PWM <v>   ESP -> PC: OK PWM <v>
 *   PC  -> ESP32: GET_STATE     ESP -> PC: STATE rpm=<r> pwm=<p> encoder=<c>
 *   PC  -> ESP32: STOP          ESP -> PC: OK STOP
 *
 * Extra commands beyond that contract (safe, additive):
 *   VERSION / INFO   -> firmware identity and the pin map, so the wiring can
 *                       always be verified from the host side. The previous
 *                       firmware was a black box, which made an encoder fault
 *                       impossible to localise.
 *   RAW              -> instantaneous A/B pin levels, for wiring diagnosis.
 *
 * Hardware: ESP32 + TB6612FNG + JGB37-520 gearmotor with quadrature hall encoder.
 */

#include <Arduino.h>

// ---------------------------------------------------------------------------
// Pin map  -- must match the physical wiring
// ---------------------------------------------------------------------------
static const int PIN_PWMA = 25;   // TB6612 PWMA
static const int PIN_AIN1 = 26;   // TB6612 AIN1
static const int PIN_AIN2 = 27;   // TB6612 AIN2
static const int PIN_ENC_A = 18;  // encoder yellow
static const int PIN_ENC_B = 19;  // encoder green
static const int PIN_STBY = -1;   // -1 = STBY hard-wired to 3V3

// ---------------------------------------------------------------------------
// Motor / encoder constants
// ---------------------------------------------------------------------------
// JGB37-520: 11 pulses per motor-shaft revolution, quadrature x4 decoding.
// GEAR_RATIO is the gearbox reduction; set it to match the actual motor.
// This unit is the 12V / 178 RPM variant: the bare 520 motor runs at roughly
// 10000 RPM, so 10000 / 178 gives a 1:56 gearbox. Verify empirically by turning
// the output shaft exactly one revolution -- the count should land near
// countsPerOutputRev() (11 * 4 * 56 = 2464).
static const float PULSES_PER_MOTOR_REV = 11.0f;
static const float QUADRATURE_MULT = 4.0f;
static float GEAR_RATIO = 56.0f;  // adjustable at runtime via SET_GEAR

static float countsPerOutputRev() {
  return PULSES_PER_MOTOR_REV * QUADRATURE_MULT * GEAR_RATIO;
}

// ---------------------------------------------------------------------------
// LEDC (PWM) configuration
// ---------------------------------------------------------------------------
static const int LEDC_CHANNEL = 0;
static const int LEDC_FREQ_HZ = 20000;  // above audible range
static const int LEDC_RESOLUTION = 8;   // 0..255, matching the host's PWM scale

// arduino-esp32 3.x replaced ledcSetup/ledcAttachPin with a pin-oriented
// ledcAttach, and ledcWrite now takes the pin instead of the channel. Wrap both
// so this sketch builds on either core version.
//
// All three driver pins get a PWM channel, because which pin carries the duty
// cycle is what selects the decay mode (see applyMotor).
#if ESP_ARDUINO_VERSION_MAJOR >= 3
#define PWM_ATTACH(pin, ch) ledcAttach((pin), LEDC_FREQ_HZ, LEDC_RESOLUTION)
#define PWM_SET(pin, ch, duty) ledcWrite((pin), (duty))
#else
#define PWM_ATTACH(pin, ch)                                     \
  do {                                                          \
    ledcSetup((ch), LEDC_FREQ_HZ, LEDC_RESOLUTION);             \
    ledcAttachPin((pin), (ch));                                 \
  } while (0)
#define PWM_SET(pin, ch, duty) ledcWrite((ch), (duty))
#endif

static const int CH_PWMA = 0;
static const int CH_AIN1 = 1;
static const int CH_AIN2 = 2;

// ---------------------------------------------------------------------------
// Encoder state
// ---------------------------------------------------------------------------
volatile long encoderCount = 0;

// Quadrature state machine. Index is (prevState << 2) | newState; the value is
// the increment. Invalid transitions map to 0 so noise cannot inject counts.
static const int8_t QUAD_TABLE[16] = {
    0, -1, 1, 0,
    1, 0, 0, -1,
    -1, 0, 0, 1,
    0, 1, -1, 0};

volatile uint8_t prevEncState = 0;

// Measured on this unit: a positive PWM produced a negative count. Whether the
// encoder counts up or down for a given rotation depends on which hall channel
// landed on A versus B, which is arbitrary. Correcting it here rather than by
// swapping wires keeps the physical rotation direction unchanged.
//
// The PID assumes a positive PWM raises the measured speed. If that is
// inverted the loop drives the error the wrong way and runs away, so this sign
// must be right before any closed-loop run.
static int encoderSign = -1;  // +1 or -1, runtime-settable via SET_ENCODER_SIGN

void IRAM_ATTR onEncoderEdge() {
  uint8_t a = (uint8_t)digitalRead(PIN_ENC_A);
  uint8_t b = (uint8_t)digitalRead(PIN_ENC_B);
  uint8_t state = (a << 1) | b;
  encoderCount += encoderSign * QUAD_TABLE[(prevEncState << 2) | state];
  prevEncState = state;
}

// ---------------------------------------------------------------------------
// Speed estimation
// ---------------------------------------------------------------------------
static long lastRpmCount = 0;
static unsigned long lastRpmMicros = 0;
static float currentRpm = 0.0f;

// Exponential smoothing: the raw difference over a short window is noisy at low speed.
static const float RPM_SMOOTH_ALPHA = 0.3f;

static void updateRpm() {
  unsigned long now = micros();
  unsigned long dtMicros = now - lastRpmMicros;
  if (dtMicros < 20000UL) {  // update at most every 20 ms
    return;
  }

  noInterrupts();
  long count = encoderCount;
  interrupts();

  long deltaCount = count - lastRpmCount;
  float dtSec = dtMicros / 1000000.0f;

  float revs = (float)deltaCount / countsPerOutputRev();
  float rpmRaw = (revs / dtSec) * 60.0f;

  currentRpm = RPM_SMOOTH_ALPHA * rpmRaw + (1.0f - RPM_SMOOTH_ALPHA) * currentRpm;

  lastRpmCount = count;
  lastRpmMicros = now;
}

// ---------------------------------------------------------------------------
// Motor drive
// ---------------------------------------------------------------------------
static int currentPwm = 0;

// Safety ceiling enforced on the device, independent of whatever the host asks
// for. The host has its own limit (REAL_PWM_MAX) but a device-side bound means
// a host bug cannot drive the motor beyond this.
//
// 200 of 255 is about 78% duty. This motor needs roughly 1.4x the PWM of the
// original testbed for the same rpm, so a 140 ceiling left the 95 rpm target
// sitting exactly at saturation. 200 restores the same headroom ratio the
// original runs had, and stays inside the motor's 0.45-1 A rating and the
// TB6612's 1.2 A continuous limit.
static int pwmLimit = 200;

// TB6612 truth table (STBY high):
//   IN1 IN2 PWM        result
//    H   L   H         forward
//    H   L   L         SHORT BRAKE
//    L   L   H         coast / stop
//
// So which pin carries the PWM decides what happens during the off-phase:
//
//   drive mode "brake": hold IN1/IN2, switch PWMA. The motor is actively
//       braked every off-period, which costs speed and costs proportionally
//       more at low duty.
//   drive mode "coast": hold PWMA high, switch IN1 (IN2 low). The off-period
//       is a freewheel, so the same duty yields a higher average speed.
//
// The original testbed's logs show ~0.96 rpm per PWM count, above what a
// 178 rpm / 12 V motor can do in brake mode, which is what this switch is for.
static bool coastDecay = false;

static void applyMotor(int pwm) {
  bool reverse = pwm < 0;
  int magnitude = abs(pwm);
  if (magnitude > pwmLimit) {
    magnitude = pwmLimit;
  }

  if (magnitude == 0) {
    PWM_SET(PIN_PWMA, CH_PWMA, 0);
    PWM_SET(PIN_AIN1, CH_AIN1, 0);
    PWM_SET(PIN_AIN2, CH_AIN2, 0);
  } else if (coastDecay) {
    // PWMA held high; duty rides on the direction pin, so the off-phase is
    // IN1=L, IN2=L, PWM=H -> coast.
    PWM_SET(PIN_PWMA, CH_PWMA, 255);
    if (reverse) {
      PWM_SET(PIN_AIN1, CH_AIN1, 0);
      PWM_SET(PIN_AIN2, CH_AIN2, magnitude);
    } else {
      PWM_SET(PIN_AIN2, CH_AIN2, 0);
      PWM_SET(PIN_AIN1, CH_AIN1, magnitude);
    }
  } else {
    // direction pins held; duty rides on PWMA, so the off-phase is a short brake.
    if (reverse) {
      PWM_SET(PIN_AIN1, CH_AIN1, 0);
      PWM_SET(PIN_AIN2, CH_AIN2, 255);
    } else {
      PWM_SET(PIN_AIN1, CH_AIN1, 255);
      PWM_SET(PIN_AIN2, CH_AIN2, 0);
    }
    PWM_SET(PIN_PWMA, CH_PWMA, magnitude);
  }

  currentPwm = reverse ? -magnitude : magnitude;
}


static void stopMotor() {
  applyMotor(0);
}

// ---------------------------------------------------------------------------
// Command watchdog: if the host stops talking, stop the motor.
// ---------------------------------------------------------------------------
static unsigned long lastCommandMillis = 0;
static const unsigned long COMMAND_TIMEOUT_MS = 1000;

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------
void setup() {
  Serial.begin(115200);


  if (PIN_STBY >= 0) {
    pinMode(PIN_STBY, OUTPUT);
    digitalWrite(PIN_STBY, HIGH);
  }

  PWM_ATTACH(PIN_PWMA, CH_PWMA);
  PWM_ATTACH(PIN_AIN1, CH_AIN1);
  PWM_ATTACH(PIN_AIN2, CH_AIN2);
  PWM_SET(PIN_PWMA, CH_PWMA, 0);
  PWM_SET(PIN_AIN1, CH_AIN1, 0);
  PWM_SET(PIN_AIN2, CH_AIN2, 0);

  // Internal pull-ups: JGB37-520 hall outputs are weak/open-collector on some
  // batches. Enabling them costs nothing when the encoder drives push-pull.
  pinMode(PIN_ENC_A, INPUT_PULLUP);
  pinMode(PIN_ENC_B, INPUT_PULLUP);

  prevEncState = (uint8_t)((digitalRead(PIN_ENC_A) << 1) | digitalRead(PIN_ENC_B));
  attachInterrupt(digitalPinToInterrupt(PIN_ENC_A), onEncoderEdge, CHANGE);
  attachInterrupt(digitalPinToInterrupt(PIN_ENC_B), onEncoderEdge, CHANGE);

  lastRpmMicros = micros();
  lastCommandMillis = millis();

  Serial.println("READY motor_controller");
  Serial.print("PINS pwma=");
  Serial.print(PIN_PWMA);
  Serial.print(" ain1=");
  Serial.print(PIN_AIN1);
  Serial.print(" ain2=");
  Serial.print(PIN_AIN2);
  Serial.print(" encA=");
  Serial.print(PIN_ENC_A);
  Serial.print(" encB=");
  Serial.println(PIN_ENC_B);
}

// ---------------------------------------------------------------------------
// Command handling
// ---------------------------------------------------------------------------
static void handleCommand(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) {
    return;
  }

  String upper = cmd;
  upper.toUpperCase();
  lastCommandMillis = millis();

  if (upper == "PING") {
    Serial.println("PONG");
    return;
  }

  if (upper == "GET_STATE") {
    noInterrupts();
    long count = encoderCount;
    interrupts();
    Serial.print("STATE rpm=");
    Serial.print(currentRpm, 3);
    Serial.print(" pwm=");
    Serial.print(currentPwm);
    Serial.print(" encoder=");
    Serial.println(count);
    return;
  }

  if (upper == "STOP") {
    stopMotor();
    Serial.println("OK STOP");
    return;
  }

  if (upper.startsWith("SET_PWM")) {
    int sp = cmd.indexOf(' ');
    if (sp < 0) {
      Serial.println("ERR missing_argument");
      return;
    }
    int value = cmd.substring(sp + 1).toInt();
    applyMotor(value);
    Serial.print("OK PWM ");
    Serial.println(currentPwm);
    return;
  }

  if (upper == "RESET_ENCODER") {
    noInterrupts();
    encoderCount = 0;
    interrupts();
    lastRpmCount = 0;
    currentRpm = 0.0f;
    Serial.println("OK RESET_ENCODER");
    return;
  }

  // Diagnostics: raw pin levels. Turning the shaft by hand must change these.
  if (upper == "RAW") {
    Serial.print("RAW a=");
    Serial.print(digitalRead(PIN_ENC_A));
    Serial.print(" b=");
    Serial.println(digitalRead(PIN_ENC_B));
    return;
  }

  if (upper == "VERSION" || upper == "INFO" || upper == "HELP") {
    Serial.println("INFO motor_controller v1 lgcontrol");
    Serial.print("PINS pwma=");
    Serial.print(PIN_PWMA);
    Serial.print(" ain1=");
    Serial.print(PIN_AIN1);
    Serial.print(" ain2=");
    Serial.print(PIN_AIN2);
    Serial.print(" encA=");
    Serial.print(PIN_ENC_A);
    Serial.print(" encB=");
    Serial.println(PIN_ENC_B);
    Serial.print("CONFIG pwm_limit=");
    Serial.print(pwmLimit);
    Serial.print(" gear_ratio=");
    Serial.print(GEAR_RATIO, 2);
    Serial.print(" counts_per_output_rev=");
    Serial.print(countsPerOutputRev(), 1);
    Serial.print(" encoder_sign=");
    Serial.print(encoderSign);
    Serial.print(" drive_mode=");
    Serial.println(coastDecay ? "COAST" : "BRAKE");
    Serial.println("CMDS PING SET_PWM STOP GET_STATE RESET_ENCODER RAW VERSION SET_GEAR SET_LIMIT SET_ENCODER_SIGN SET_DRIVE_MODE");
    return;
  }

  if (upper.startsWith("SET_GEAR")) {
    int sp = cmd.indexOf(' ');
    if (sp < 0) {
      Serial.println("ERR missing_argument");
      return;
    }
    float value = cmd.substring(sp + 1).toFloat();
    if (value <= 0.0f) {
      Serial.println("ERR invalid_gear_ratio");
      return;
    }
    GEAR_RATIO = value;
    Serial.print("OK GEAR ");
    Serial.println(GEAR_RATIO, 2);
    return;
  }

  if (upper.startsWith("SET_DRIVE_MODE")) {
    int sp = cmd.indexOf(' ');
    if (sp < 0) {
      Serial.println("ERR missing_argument");
      return;
    }
    String mode = cmd.substring(sp + 1);
    mode.trim();
    mode.toUpperCase();
    if (mode == "COAST") {
      coastDecay = true;
    } else if (mode == "BRAKE") {
      coastDecay = false;
    } else {
      Serial.println("ERR invalid_drive_mode");
      return;
    }
    applyMotor(currentPwm);
    Serial.print("OK DRIVE_MODE ");
    Serial.println(coastDecay ? "COAST" : "BRAKE");
    return;
  }

  if (upper.startsWith("SET_ENCODER_SIGN")) {
    int sp = cmd.indexOf(' ');
    if (sp < 0) {
      Serial.println("ERR missing_argument");
      return;
    }
    int value = cmd.substring(sp + 1).toInt();
    if (value != 1 && value != -1) {
      Serial.println("ERR invalid_sign");
      return;
    }
    encoderSign = value;
    Serial.print("OK ENCODER_SIGN ");
    Serial.println(encoderSign);
    return;
  }

  if (upper.startsWith("SET_LIMIT")) {
    int sp = cmd.indexOf(' ');
    if (sp < 0) {
      Serial.println("ERR missing_argument");
      return;
    }
    int value = cmd.substring(sp + 1).toInt();
    if (value < 0 || value > 255) {
      Serial.println("ERR invalid_limit");
      return;
    }
    pwmLimit = value;
    if (abs(currentPwm) > pwmLimit) {
      applyMotor(currentPwm);  // re-clamp immediately
    }
    Serial.print("OK LIMIT ");
    Serial.println(pwmLimit);
    return;
  }

  Serial.println("ERR unknown_cmd");
}

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------
void loop() {
  updateRpm();

  while (Serial.available() > 0) {
    String line = Serial.readStringUntil('\n');
    handleCommand(line);
  }

  // Fail safe: a silent host must not leave the motor running.
  if (currentPwm != 0 && (millis() - lastCommandMillis) > COMMAND_TIMEOUT_MS) {
    stopMotor();
    Serial.println("EVENT command_timeout_stop");
  }
}
