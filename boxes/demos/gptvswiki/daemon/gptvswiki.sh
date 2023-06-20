#!/bin/bash

# GPTvsWiki daemon script

# Check if GPTvsWiki client is running
if pidof "python3" > /dev/null
then
  echo "Python (GPTvsWiki) is running."
else
  echo "Python (GPTvsWiki) is not running...starting"
  cd /home/kampff/NoBlackBoxes/repos/OtherBlackBoxes/boxes/websites/applications/flask/gptvswiki
  python3 app.py
fi

# Finish
exit 0
#FIN