#!/bin/bash

# Hello NB3 daemon script

# Check if Hello NB3 client is running
if pidof "python3" > /dev/null
then
  echo "Python (Hello NB3) is running."
else
  echo "Python (Hello NB3) is not running...starting"
  cd /home/kampff/NoBlackBoxes/repos/OtherBlackBoxes/boxes/demos/hello_nb3
  export DISPLAY=:0
  python3 hello_nb3.py
fi

# Finish
exit 0
#FIN