#!/bin/bash

# Chromium daemon script

# Check if Chromium browser is running
if pidof "chromium-browse" > /dev/null
then
  echo "Chromium is running."
else
  echo "Chromium is not running...starting"
  export DISPLAY=:0
  sed -i 's/"exited_cleanly":false/"exited_cleanly":true/' /home/kampff/.config/chromium/Default/Preferences
  sed -i 's/"exit_type":"Crashed"/"exit_type":"Normal"/' /home/kampff/.config/chromium/Default/Preferences
  echo "Chromium starting in fullscreen now!"
  /usr/bin/chromium-browser --window-size=1920,1080 --window-position=0,0 --start-fullscreen http://127.0.0.1:5000
  echo "Chromium should be running!"
fi

# Finish
exit 0
#FIN