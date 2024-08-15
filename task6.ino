/*
* Team Id: 1714
* Author List: Shreyas, Neeraj, Srujan, Aparna
* Filename: Task5A.ino
* Theme: Geo Guide
* Functions: ClearPath,ClearDistance,ClearVisit,ClearTurns,ClearParent,Dijikstra,detectEvents,path_plan,initWifi,lineAndLEDSetup,readSensor,getError,calculatePID,motorPIDcontrol,stop,left_turn,right_turn,forward,reverse,lineFollow,wifi,take_turn,setup,loop
* Global Variables: turnOrientation, path,distance,parent,visited,all_envents_path,ai,aidx,all_events_turn,aei,graph,adj,prev,curr,next,aind,event,No_Events
*/

#include <WiFi.h>
#include <WiFiClient.h>

#define MAX 100 

// Constants for orientation of Vanguard

#define LEFT 0  
#define RIGHT 1
#define FORWARD 2
#define BACKWARD 3
/*
line followng constants
*/
#define en1 19  //Enable for left motor
#define en2 18  //Enable for right motor


// Pin configuration for motors
#define r1 15 
#define r0 2
#define l0 4
#define l1 16


//Pin configuration for IR sensors
#define R 14
#define C 12
#define L 13
#define ER 26
#define EL 27


//PWM channels for motors and buzzer
#define ch1 3
#define ch2 2
#define ch3 15
#define green 33 
#define buzzer 21

// Variable Name: turnOrientation
// Description: This 2D array stores the possible orientations after turning in each direction.
//              Each row corresponds to a current orientation, and each column corresponds
//              to a possible turning direction. The values represent the new orientation
//              after turning: FORWARD, BACKWARD, LEFT, RIGHT.
// Expected Range of Values: Each element of the array should be one of the predefined
//                           constants: FORWARD, BACKWARD, LEFT, RIGHT.


int turnOrientation[][4] = { { BACKWARD,
                               FORWARD,
                               LEFT,
                               RIGHT },
                             { FORWARD,
                               BACKWARD,
                               RIGHT,
                               LEFT },
                             { RIGHT,
                               LEFT,
                               BACKWARD,
                               FORWARD },
                             { LEFT,
                               RIGHT,
                               FORWARD,
                               BACKWARD } };

int path[17];  // Stores Graph through which bot traverse
int distance[17]; // Stores distances between the nodes of the graph  
int parent[17]; // Stores previous node of every node
int visited[17]; // Stores the visited flag for the nodes
int all_envents_path[90]; // Stores combined path for all the set of start and end nodes 
int ai = 0, aidx = 0;  //for take turn in run
int all_events_turn[90]; // Stores the turn tobe taken on each node
int aei = 0;

// Variable Name: graph
// Description: This 2D array represents a graph where each element (graph[i][j])
//              denotes the weight or distance between nodes i and j. A value of 0
//              indicates no direct connection between the nodes. The indices of the
//              array correspond to the nodes in the graph. For example, graph[0][1]
//              represents the weight from node 0 to node 1.
// Expected Range of Values: Non-negative integers representing distances or weights
//                           between nodes. A value of 0 indicates no connection.


int graph[17][17] = {           
  { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },  
  { 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
  { 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
  { 0, 0, 1, 0, 0, 1, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
  { 0, 1, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 },
  { 0, 0, 0, 1, 2, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 },
  { 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
  { 0, 0, 0, 3, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 },
  { 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0 },
  { 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0 },
  { 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0 },
  { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0 },
  { 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0 },
  { 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 2 },
  { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 2, 0 },
  { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 3 },
  { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 3, 0 }
};

// Variable Name: adj
// Description: This 2D array represents the adjacency matrix of a graph where each
//              row corresponds to a node in the graph and each column corresponds
//              to a direction (left, right, forward, backward). The values in the
//              array represent the indices of the neighboring nodes in the specified
//              direction. A value of -1 indicates no neighbor in that direction.
//              The directions are denoted as follows: 'l' for left, 'r' for right,
//              'f' for forward, and 'b' for backward.
// Expected Range of Values: Integers representing indices of neighboring nodes or -1
//                           to denote absence of a neighbor.

int adj[17][4] = {
  // l    r    f    b
  { -1, -1, 1, -2 },   // Node 0
  { -1, 2, 4, 0 },     // Node 1
  { 1, 3, -1, -1 },    // Node 2
  { 2, 7, 5, -1 },     // Node 3
  { -1, 5, 8, 1 },     // Node 4
  { 4, 6, 10, 3 },     // Node 5
  { 5, 7, -1, -1 },    // Node 6
  { 6, -1, 12, 3 },    // Node 7
  { -1, 9, 13, 4 },    // Node 8
  { 8, 10, -1, -1 },   // Node 9
  { 9, 11, 14, 5 },    // Node 10
  { 10, 12, -1, -1 },  // Node 11
  { 11, -1, 15, 7 },   // Node 12
  { -1, 14, 16, 8 },   // Node 13
  { 13, 15, -1, 10 },  // Node 14
  { 14, -1, 16, 12 },  // Node 15
  { 13, 15, -1, -1 }   // Node 16
};

// Variable Name: prev
// Description: Previous node index in a sequence or loop.

// Variable Name: curr
// Description: Current node index in a sequence or loop, initialized to -2.

// Variable Name: next
// Description: Next node index in a sequence or loop.

// Variable Name: aind
// Description: Index used for some purpose in the context of the program, initially 0.

int prev, curr = -2, next, aind = 0;

// Variable Name: event
// Description: This 2D array stores the start and end positions for each event location.
//              Each row corresponds to an event, and each column stores the start and end
//              positions respectively.
// Expected Range of Values: Integers representing positions for start and end of each event.

int event[5][2]; 
int No_Events = -1;


// Variable Name: ssid
// Description: SSID (network name) for WiFi connection.
// Expected Range: String representing the SSID.

const char *ssid = "GG_1714";

// Variable Name: password
// Description: Password for WiFi connection.
// Expected Range: String representing the WiFi password.

const char *password = "123456789";

// Variable Name: serverPort
// Description: Port number for the server.
// Expected Range: Integer representing the port number.

const int serverPort = 65534;

const char *static_ip = "192.168.231.108";  // Set your desired static IP address
const char *gateway = "192.168.231.1";      // Set your router's IP address
const char *subnet = "255.255.255.0";      // Set your subnet mask

// Object Name: server
// Description: WiFi server object for communication.
// Constructor Input: Integer representing the server port number.

WiFiServer server(serverPort);

String event_string = "";
int mode = 0;
int toggle = 1;

// Variable Name: sensor
// Description: Array storing sensor readings.
// Expected Range: Array of integers representing sensor readings.

int sensor[5] = { 0, 0, 0, 0, 0 };
float currentError = 0;
int previousError = 0;

/* 
 * Function Name: ClearPath 
 * Input:         None 
 * Output:        None 
 * Logic:         This function clears the elements of the 'path' array by 
 *                setting each element to -1. It iterates through the 'path' 
 *                array using a for loop and assigns -1 to each element. 
 * Example Call:  ClearPath(); 
 */ 

void ClearPath() {
  for (int i = 0; i < 17; i++) {
    path[i] = -1;
  }
}

/* 
 * Function Name: ClearDistance
 * Input:         None
 * Output:        None
 * Logic:         Resets each element of the 'distance' array to the maximum value.
 *                This is typically used to clear/reset distance values in a graph algorithm.
 * Example Call:  ClearDistance();
 */

void ClearDistance() {
  for (int i = 0; i < 17; i++) {
    distance[i] = MAX;
  }
}

void ClearVisit() {
  for (int i = 0; i < 17; i++) {
    visited[i] = 0;
  }

}

void ClearTurns() {
  for (int i = 0; i < 90; i++) {
    all_events_turn[i] = -1;
  }
}

void ClearParent() {
  for (int i = 0; i < 17; i++) {
    parent[i] = 18;
  }
}
/*
* Function Name: Dijikstra
* Input: start - The starting node for the shortest path calculation
*        end   - The ending node for the shortest path calculation
* Output: The length of the shortest path from the start node to the end node
* Logic: This function implements Dijkstra's algorithm to find the shortest path
*        between two nodes in a graph. It iteratively explores the graph starting
*        from the given start node and updates the distance of each node from the
*        start node until it reaches the end node. It keeps track of the shortest
*        path using parent pointers.
* Example Call: int pathLength = Dijikstra(startNode, endNode);
*/
int Dijikstra(int start, int end) {
  ClearDistance();
  ClearVisit();
  ClearParent();
  distance[start] = 0;
  parent[start] = -1;

  for (int i = 0; i < 17; i++) {
    int i_min = -1, d_min = MAX;
    for (int j = 0; j < 17; j++) {
      if (visited[j] == 0 && distance[j] < d_min) {
        d_min = distance[j];
        i_min = j;
      }
    }
    visited[i_min] = 1;
    for (int j = 0; j < 17; j++) {
      if (visited[j] == 0 && graph[i_min][j] != 0) {
        if (graph[i_min][j] + distance[i_min] < distance[j]) {
          distance[j] = graph[i_min][j] + distance[i_min];
          parent[j] = i_min;
        }
      }
    }
  }
  ClearPath();
  int i = 0;
  while (end != -1 && i < 17) {
    path[i] = end;
    end = parent[end];
    i++;
  }
  return i;  //i stores the number of nodes to be visited
}

/*
  * Function Name: detectEvents
  * Input: s - String containing event information in the format 'xxyy', where 'xx' is the number of events
  *             and 'yy' are the event nodes represented as characters ('a' to 'q' for nodes 0 to 16)
  * Output: None
  * Logic: This function parses the input string to extract event information. It first reads the number of events
  *        from the first character of the string. Then, it extracts the event nodes from the subsequent characters
  *        of the string and stores them in the 'event' array for further processing.
  * Example Call: detectEvents("2ab");
  */
void detectEvents(String s) {
  No_Events = s[0] - 'a' + 1;
  for (int i = 1; i <= No_Events; i++) {
    event[i - 1][1] = s[i] - 'a' + 1;
  }
  event[0][0] = 0;
  for (int i = 1; i <= No_Events + 1; i++) {
    event[i][0] = event[i - 1][1];
  }
}
/*
  * Function Name: path_plan
  * Input: None
  * Output: None
  * Logic: This function plans the path to navigate through the graph based on detected events. It uses Dijkstra's algorithm
  *        to find the shortest path between each pair of consecutive events. The function iterates over each pair of events,
  *        calculates the shortest path between them using Dijkstra's algorithm, and stores the nodes of the path in the 'all_envents_path'
  *        array. It also determines the appropriate turns (LEFT, RIGHT, FORWARD, BACKWARD) at each node based on the adjacency
  *        matrix and stores them in the 'all_events_turn' array.
  * Example Call: path_plan();
  */

void path_plan() {
  for (int eve = 0; eve <= No_Events; eve++) {

    int n = Dijikstra(event[eve][0], event[eve][1]);
    for (int i = n - 1; i > 0; i--) {
      all_envents_path[ai++] = path[i];
    }
  }
  for (aind = 0; aind < ai; aind++) {
    prev = curr;
    curr = all_envents_path[aind];
    next = all_envents_path[aind + 1];

    int turnSelector, turn;
    for (int j = 0; j < 4; j++) {
      if (adj[curr][j] == prev) {
        turnSelector = j;
      }
      if (adj[curr][j] == next) {
        turn = j;
      }
    }
    int t = turnOrientation[turnSelector][turn];
    all_events_turn[aind] = t;
    aei++;
  }
}

/* 
 * Function Name: initWifi
 * Input:         None
 * Output:        None
 * Logic:         Initializes WiFi connection, sets static IP configuration,
 *                and begins the server.
 * Example Call:  initWifi();
 */

void initWifi() {
  digitalWrite(green, HIGH);
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.println("Connecting to WiFi..");
  }


  Serial.println("Connected to the WiFi network");
  digitalWrite(green, LOW);
  // Set static IP configuration
  IPAddress local_ip(192, 168, 231, 108);    // Convert static_ip to four integers
  IPAddress gateway_ip(192, 168, 231, 1);    // Convert gateway to four integers
  IPAddress subnet_mask(255, 255, 255, 0);  // Convert subnet to four integers
  WiFi.config(local_ip, gateway_ip, subnet_mask);

  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());
  server.begin();
}

/* 
 * Function Name: lineAndLEDSetup
 * Input:         None
 * Output:        None
 * Logic:         Sets up pins for line sensors, LEDs, and buzzer.
 * Example Call:  lineAndLEDSetup();
 */

void lineAndLEDSetup() {
  ledcSetup(ch1, 25000, 8);
  ledcSetup(ch2, 25000, 8);
  ledcSetup(ch3, 2000, 8);
  ledcAttachPin(en1, ch1);
  ledcAttachPin(en2, ch2);
  ledcAttachPin(buzzer, ch3);
  pinMode(L, INPUT);
  pinMode(C, INPUT);
  pinMode(R, INPUT);
  pinMode(EL, INPUT);
  pinMode(ER, INPUT);
  pinMode(l0, OUTPUT);
  pinMode(l1, OUTPUT);
  pinMode(r0, OUTPUT);
  pinMode(r1, OUTPUT);
  pinMode(green, OUTPUT);
}

/* 
 * Function Name: readSensor
 * Input:         None
 * Output:        None
 * Logic:         Reads digital values from line sensors and updates the 'sensor' array.
 * Example Call:  readSensor();
 */

void readSensor() {
  sensor[0] = digitalRead(EL);
  sensor[1] = digitalRead(L);
  sensor[2] = digitalRead(C);
  sensor[3] = digitalRead(R);
  sensor[4] = digitalRead(ER);
}



void lineSensorParallel(void *params) {
  while (1) {
    readSensor();
    // vTaskDelay(50/portTICK_RATE_MS);
  }
}

// Variable Name: Kp
// Description: Proportional gain coefficient.
// Expected Range: Float representing the proportional gain value.

float Kp = 30;
float PIDvalue = 0;
int base_l = 195; // Speed for left motor
int base_r = 185; // Spped for right motor

/*
* Function Name: getError
* Input: None
* Output: None
* Logic: This function determines the error value based on the sensor readings from the line-following array. It interprets
*        the sensor readings to identify the current position of the robot relative to the line and calculates the appropriate
*        error value accordingly. The error value represents the deviation of the robot's position from the desired path.
* Example Call: getError();
*/

void getError() {

  if (sensor[0] == 0 && sensor[1] == 1 && sensor[2] == 1 && sensor[3] == 1 && sensor[4] == 0) {
    // l c r = 1 1 1
    base_l = 199;
    base_r = 190;
    mode = 7;  //increment the index for next node and take turns;
    currentError = 0;
    return;
  }
  if (sensor[0] == 1 && sensor[1] == 0 && sensor[2] == 0 && sensor[3] == 0 && sensor[4] == 0) {
    // el = 1
    currentError = 1.6;
  } else if (sensor[0] == 0 && sensor[1] == 1 && sensor[2] == 0 && sensor[3] == 0 && sensor[4] == 0) {
    // l = 1
    currentError = -0.5;
  } else if (sensor[0] == 0 && sensor[1] == 1 && sensor[2] == 1 && sensor[3] == 0 && sensor[4] == 0) {
    // lc = 1
    currentError = -0.5;
  } else if (sensor[0] == 0 && sensor[1] == 0 && sensor[2] == 1 && sensor[3] == 0 && sensor[4] == 0) {
    // c = 1
    currentError = 0;
  } else if (sensor[0] == 0 && sensor[1] == 0 && sensor[2] == 1 && sensor[3] == 1 && sensor[4] == 0) {
    // cr = 1
    currentError = 0.5;
  } else if (sensor[0] == 0 && sensor[1] == 0 && sensor[2] == 0 && sensor[3] == 1 && sensor[4] == 0) {
    // r = 1
    currentError = 0.5;
  } else if (sensor[0] == 0 && sensor[1] == 0 && sensor[2] == 0 && sensor[3] == 0 && sensor[4] == 1) {
    // er = 1
    currentError = -1.6;
  } else if (sensor[0] == 0 && sensor[1] == 0 && sensor[2] == 0 && sensor[3] == 0 && sensor[4] == 0) {
    // all white
    currentError = 0;
  } else if ((sensor[0] == 1 || sensor[4] == 1) && sensor[1] == 1) {
    currentError = 0;
  } else if ((sensor[0] == 1 || sensor[4] == 1) && sensor[2] == 1) {
    currentError = 0;
  } else if ((sensor[0] == 1 || sensor[4] == 1) && sensor[3] == 1) {
    currentError = 0;
  }
}
/*
* Function Name: calculatePID
* Input: None
* Output: None
* Logic: This function calculates the PID (Proportional-Integral-Derivative) value based on the current error and previous error values.
*        It uses the error value calculated by the getError function to determine the correction needed for the robot's movement.
*        The PID value is calculated using the proportional, integral, and derivative terms, weighted by their respective coefficients.
* Example Call: calculatePID();
*/
void calculatePID() {
  float P = currentError;
  PIDvalue = (Kp * P);
  previousError = currentError;
}
/*
* Function Name: motorPIDcontrol
* Input: None
* Output: None
* Logic: This function adjusts the motor speeds based on the calculated PID value. It modifies the motor speeds to correct
*        the robot's position relative to the line by applying the PID control algorithm. The PID value, computed by the
*        calculatePID function, is used to adjust the motor speeds proportionally to minimize the error and maintain the
*        robot's trajectory along the line.
* Example Call: motorPIDcontrol();
*/

void motorPIDcontrol() {
  int leftMotorSpeed = base_l + (float)PIDvalue;
  int rightMotorSpeed = base_r - (float)PIDvalue;
  leftMotorSpeed = min(leftMotorSpeed, 255);
  rightMotorSpeed = min(rightMotorSpeed, 255);
  ledcWrite(ch1, leftMotorSpeed);
  ledcWrite(ch2, rightMotorSpeed);

  digitalWrite(l0, LOW);
  digitalWrite(l1, HIGH);
  digitalWrite(r0, LOW);
  digitalWrite(r1, HIGH);
}

/* 
 * Function Name: stop
 * Input:         None
 * Output:        None
 * Logic:         Stops the motors by setting appropriate PWM values and digital outputs.
 * Example Call:  stop();
 */

void stop() {
  ledcWrite(ch1, 180);
  ledcWrite(ch2, 172);
  digitalWrite(l0, LOW);
  digitalWrite(l1, LOW);
  digitalWrite(r0, LOW);
  digitalWrite(r1, LOW);
}

/* 
 * Function Name: left_turn
 * Input:         None
 * Output:        None
 * Logic:         Turns the robot left by setting appropriate PWM values and digital outputs.
 * Example Call:  left_turn();
 */

void left_turn() {
  ledcWrite(ch1, 189);
  ledcWrite(ch2, 193);
  digitalWrite(l0, LOW);
  digitalWrite(l1, LOW);
  digitalWrite(r0, LOW);
  digitalWrite(r1, HIGH);
}

/* 
 * Function Name: left_turn
 * Input:         None
 * Output:        None
 * Logic:         Turns the robot right by setting appropriate PWM values and digital outputs.
 * Example Call:  left_turn();
 */

void right_turn() {
  ledcWrite(ch1, 195);
  ledcWrite(ch2, 193);
  digitalWrite(l0, LOW);
  digitalWrite(l1, HIGH);
  digitalWrite(r0, LOW);
  digitalWrite(r1, LOW);
}

/* 
 * Function Name: forward
 * Input:         None
 * Output:        None
 * Logic:         Moves the robot forward by setting appropriate PWM values and digital outputs.
 *                It also blinks the green LED for a brief period to indicate forward motion.
 * Example Call:  forward();
 */

void forward() {
  ledcWrite(ch1, 190);
  ledcWrite(ch2, 184);
  digitalWrite(l0, LOW);
  digitalWrite(l1, HIGH);
  digitalWrite(r0, LOW);
  digitalWrite(r1, HIGH);
  digitalWrite(green,HIGH);
  delay(50);
  digitalWrite(green,LOW);
}

/* 
 * Function Name: reverse
 * Input:         None
 * Output:        None
 * Logic:         Moves the robot backward by setting appropriate PWM values and digital outputs.
 *                It delays for a specific duration to control the reverse motion.
 * Example Call:  reverse();
 */

void reverse() {
  ledcWrite(ch1, 188);
  ledcWrite(ch2, 186);
  digitalWrite(l0, LOW);
  digitalWrite(l1, HIGH);
  digitalWrite(r0, HIGH);
  digitalWrite(r1, LOW);
  delay(1280);
}

/* 
 * Function Name: lineFollow
 * Input:         None
 * Output:        None
 * Logic:         Performs line following by obtaining the error, calculating PID values,
 *                and controlling the motors using PID.
 * Example Call:  lineFollow();
 */

void lineFollow() {
  getError();
  calculatePID();
  motorPIDcontrol();
}

/*
* Function Name: wifi
* Input: params - Pointer to input parameters (not used in this function)
* Output: None
* Logic: This function handles WiFi client connections and processes incoming signals from connected clients. It listens
*        for incoming connections on the specified server port and reads data sent by clients. Depending on the received
*        signals, it updates the mode variable to control the behavior of the robot, such as path planning, node traversal,
*        stopping, or other actions.
* Example Call: wifi(NULL);
*/

void wifi(void *params) {
  while (1) {
    WiFiClient client = server.available();
    if (client) {
      Serial.println("New client connected");
      while (client.connected()) {
        if (client.available()) {
          String signal = client.readStringUntil('\n');
          Serial.println(" ");
          Serial.println(signal);
          if (signal.equals("ZONE")) {
            mode = 5;
            //stop event
          } else if (signal.equals("NODE")) {
            mode = 7;
          } else if (signal.equals("STOP")) {
            mode = 6;
            //stop the bot
          } else if (signal[0] == 'z') {
            event_string = signal.substring(1);
            mode = 1;
            //plan the path
          }
        }
      }
      client.stop();
      Serial.println("Client disconnected");
    }
  }
}

/* 
 * Function Name: take_turn
 * Input:         Integer representing the type of turn (0 for left, 1 for right, 2 for forward, 3 for reverse)
 * Output:        None
 * Logic:         Executes the specified turn based on the input value using a switch-case statement.
 * Example Call:  take_turn(turn);
 */

void take_turn(int turn) {
  switch (turn) {
    case 0:
      left_turn();
      break;
    case 1:
      right_turn();
      break;
    case 2:
      forward();
      break;
    case 3:
      reverse();
      break;
  }
}

/* 
 * Function Name: setup
 * Input:         None
 * Output:        None
 * Logic:         Initializes serial communication, sets up line sensors, LED, buzzer, and WiFi.
 *                Creates tasks for WiFi communication and line sensor reading.
 * Example Call:  Automatically called by the Arduino framework at the beginning of the program.
 */

void setup() {
  Serial.begin(115200);
  lineAndLEDSetup();
  ledcWriteTone(ch3, 2500);
  initWifi();
  xTaskCreate(
    wifi,
    "Wifi listner",
    10000,
    NULL,
    tskIDLE_PRIORITY,
    NULL);

  xTaskCreate(
    lineSensorParallel,
    "Line sensor read",
    10000,
    NULL,
    tskIDLE_PRIORITY,
    NULL);
}

/* 
 * Function Name: loop
 * Input:         None
 * Output:        None
 * Logic:         Executes different modes of operation based on the 'mode' variable.
 *                Handles path planning, line following, stopping at nodes, taking turns,
 *                and event handling.
 * Example Call:  Automatically called by the Arduino framework repeatedly after setup.
 */

void loop() {
  switch (mode) {
    case 0:  //waiting for signal
      break;
    case 1:  //path planning
      detectEvents(event_string);
      delay(100);
      path_plan();
      delay(100);
      mode = 2;  //after path planning
      break;
    case 2:  // line follow
      lineFollow();
      break;
    case 7:  //increment the node
      aidx++;
      stop();
      delay(0);
      mode = 3;
      break;
    case 3:  //stop on node take turn
      if (all_events_turn[aidx + 1] != -1) {
        if (all_events_turn[aidx] == 2) {
          if (sensor[1] == 0 || sensor[2] == 0 || sensor[3] == 0) {
            mode = 2;
          } else
            forward();
        } else if (all_events_turn[aidx] == 3) {
          reverse();
          mode = 2;
        } else if (sensor[1] == 0 && sensor[2] == 1 && sensor[3] == 0) {
          mode = 2;  // line follow
        } else {
          take_turn(all_events_turn[aidx]);
        }
      } else {
        mode = 6;
      }
      break;
    case 5:  // when bot reach on event zone
      stop();
      delay(200);
      ledcWriteTone(ch3, 0);
      delay(1000);
      ledcWriteTone(ch3, 2500);
      mode = 7;
      break;
    case 6:  // end signals
      stop();
      ledcWriteTone(ch3, 0);
      delay(5000);
      ledcWriteTone(ch3, 2500);
      Serial.println("End or run.....");
      delay(1000);
      mode = 69;
      break;
    case 69:
      break;
  }
}