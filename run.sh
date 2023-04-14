#!/bin/bash

# Change to the server directory and run the first command in the background
cd ReversiServer
java Reversi 3 &
cd ..
sleep 2  # Wait for 2 seconds

# Change to the BL1M3Y directory and run the second command in the background
java -jar BL1M3Y.jar localhost 1 <<< "0" &
sleep 2  # Wait for 2 seconds

# Change to the glitch directory and run the third command in the background
python client_ai.py localhost 2 &
