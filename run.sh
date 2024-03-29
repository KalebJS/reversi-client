#!/bin/bash

# Change to the server directory and run the first command in the background
cd ReversiServer
java Reversi 3 &
cd ..
sleep 1  # Wait for 2 seconds

# Change to the BL1M3Y directory and run the second command in the background
java -jar BL1M3Y.jar localhost 1 <<< "1" &
sleep 1  # Wait for 2 seconds

# Change to the glitch directory and run the third command in the background
python -m cProfile -o result.prof client_ai.py localhost 2 &
