/*
  Serial Controller
    - Respond to single character commands received via serial with servo motion
*/

#include <Servo.h>    // This includes the "servo" library
Servo left, right;    // This creates to servo objects, one for each motor
bool state_l = false;
bool state_r = false;
bool led_l = false;
bool led_r = false;
int interval = 10;
int count = 0;

// Declare control functions
void forward();
void backward();
void turn_left();
void turn_right();
void stop();

void setup() {
  // Initialize serial port
  Serial.begin(19200);

  // Attach servo pins
  right.attach(9);    // Assign right servo to digital (PWM) pin 9 (change accorinding to your connection)
  left.attach(10);    // Assign left servo to digital (PWM) pin 10 (change accorinding to your connection)

  // Initialize (no motion)
  left.write(90);
  right.write(90);

  // Initialize LED output pins
  pinMode(12, OUTPUT);
  pinMode(13, OUTPUT);
}

void loop() {
  
  // Check for any incoming bytes
  if (Serial.available() > 0) {
    char newChar = Serial.read();

    // Respond to command "f"
    if(newChar == 'f') {
      forward();
    }

    // Respond to command "b"
    if(newChar == 'b') {
      backward();
    }

    // Respond to command "l"
    if(newChar == 'l') {
      turn_left();
    }

    // Respond to command "r"
    if(newChar == 'r') {
      turn_right();
    }

    // Respond to command "x"
    if(newChar == 'x') {
      stop();
    }

    // Respond to command "h" - hearing
    if(newChar == 'h') {
      led_l = true;
      led_r = false;
    }

    // Respond to command "s" - speaking
    if(newChar == 's') {
      led_l = false;
      led_r = true;
    }

    // Respond to command "w" - waiting
    if(newChar == 'w') {
      led_l = false;
      led_r = false;
    }
  }

  // Toggle LEDs
  if(led_l)
  {
    digitalWrite(12, state_l);
  }
  else
  {
    digitalWrite(12, false);
  }
  if(led_r)
  {
    digitalWrite(13, state_r);
  }
  else
  {
    digitalWrite(13, false);
  }

  // Toggle state
  if((count % interval) == 0)
  {
    state_l = !state_l;
    state_r = !state_r;
  }

  // Wait a bit
  delay(10);

  // Increment counter
  count = count + 1;
}

// Forward
void forward()
{
  left.write(180);
  right.write(0);
}

// Backward
void backward()
{
  left.write(0);
  right.write(180);
}

// Left
void turn_left()
{
  left.write(0);
  right.write(0);
}

// Right
void turn_right()
{
  left.write(180);
  right.write(180);
}

// Stop
void stop()
{
  left.write(90);
  right.write(90);
}