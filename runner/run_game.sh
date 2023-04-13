#!/bin/bash

# Change to the server directory and run the first command in the background
cd server
java Reversi 10 &> ../serverOutput.txt &
cd ..
sleep 2  # Wait for 2 seconds


# Change to the BL1M3Y directory and run the second command in the background
cd BL1M3Y
java -jar BL1M3Y.jar localhost 1 <<< "0" &
cd ..
sleep 2  # Wait for 2 seconds

# Change to the glitch directory and run the third command in the background
cd glitch
python client_ai.py localhost 2 &
cd ..