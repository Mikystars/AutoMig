#include <Servo.h>
#include <Adafruit_NeoPixel.h>
// Amarillo ESC
// Gris Servo
// Data (Amarillo) al pin 6
// VCC (Rojo) provee 5v a ningun lado o VIN
// GND (Marron) a GND
// Neopixels al pin 9

/*
 * Lee un código desde serial
 * Para acelerar: VXX (XX valor entre 40 y 60)
 * Para girar: GXXXX.
 * G1550 = 90º Centro
 * G1200 = 0º Izquierda
 * G1900 = 180º Derecha
 * 
 */


#define ESC_PIN  5
#define SERVO_PIN  6
#define TRIGGER_PIN 2
#define ECHO_PIN 3
#define NEOPIXEL_PIN 9
#define NUMPIXELS 16

Adafruit_NeoPixel pixels(NUMPIXELS, NEOPIXEL_PIN, NEO_GRB + NEO_KHZ800);
Servo ESC;
Servo direccion;

int vel = 35;
int dir = 1550;

String readString;

void setup() {
  //Inicio comunicacion serial
  //Serial.begin(500000);
  Serial.begin(250000);
  //Serial.begin(115200);

  // Los objetos de los motores a sus pines
  ESC.attach(ESC_PIN);  // Controlador de velocidad a pin 5
  direccion.attach(SERVO_PIN);  // Servo a pin 6

  // Inicilizo los constroles, 90 gradosde giro y parado
  ESC.write(vel);
  direccion.writeMicroseconds(dir);

  // Inicializo el sensor de ultrasonido
  pinMode(TRIGGER_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  digitalWrite(TRIGGER_PIN, LOW);

  // Inicializo la tira de leds
  pixels.begin();
  //pixels.clear();

  // Todos los pixels a un color
  for(int i = 0; i < NUMPIXELS; i++){
    pixels.setPixelColor(i, pixels.Color(40, 20, 20));
  }
  pixels.show();
  
}

unsigned long prevMillis = 0;
const long wait = 250;

void loop() {
  unsigned long currMillis = millis();
  int dist = 100;

  if (currMillis - prevMillis >= wait) {
    //dist = scan();
    prevMillis = currMillis;
  }
  
  // Si hay algo a menos de 30cm, PARAR
  if (dist < 15){
    ESC.write(0);
  }

  //Serial.print("Distancia: ");
  //Serial.print(d);
  //Serial.print("cm");
  //Serial.println();
  

  String buff; // Buffer de la lectura serial
  
  if (Serial.available())  {
    char c = Serial.read();  //lee un byte desde serial
    
    if (c == '\n') {
      if (readString.length() > 0) {
        if (readString.startsWith("V")){
          for(int i = 1; i <= readString.length(); i++){
            buff += readString[i];
          }
          vel = buff.toInt();

          //Serial.print(buff + " : "); //Debug

          if(vel >= 0 && vel <= 70){
            ESC.write(vel);
          }
        }

        else if (readString.startsWith("G")){
          for(int i = 1; i <= readString.length(); i++){
            buff += readString[i];
          }
          dir = buff.toInt();
          
          //Serial.print(buff + " : "); //Debug

          if(dir >= 1200 && dir <= 1900){
            direccion.writeMicroseconds(dir);
          }
        }
        
        //Serial.println(readString); //Debug
        readString=""; //Vacía la variable para la siguiente entrada
      }
    }
    
    else {     
      readString += c; //Agrega el byte leido al final
    }
  }
}

int scan(){
  long t; // Timepo que tarda en llegar el eco
  long d; // Distancia en centimetros
  
  digitalWrite(TRIGGER_PIN, HIGH);
  delayMicroseconds(10);          // Pulso de 10us
  digitalWrite(TRIGGER_PIN, LOW);

  t = pulseIn(ECHO_PIN, HIGH); // Obtengo el ancho del pulso
  d = t/59;                // Ecalo el tiempo a una distancia en cm

  return d;
}
