#!/bin/bash
set -e

sudo systemctl unmask hello_nb3

sudo cp "/home/kampff/NoBlackBoxes/repos/OtherBlackBoxes/boxes/demos/hello_nb3/daemon/hello_nb3.timer" "/etc/systemd/system/hello_nb3.timer"

sudo cp "/home/kampff/NoBlackBoxes/repos/OtherBlackBoxes/boxes/demos/hello_nb3/daemon/hello_nb3.service" "/etc/systemd/system/hello_nb3.service"

sudo systemctl enable hello_nb3.timer

# Finish
echo "..."
echo "FIN"
exit 0
#FIN
