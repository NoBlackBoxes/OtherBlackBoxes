#!/bin/bash
set -e

sudo systemctl unmask chromium
sudo systemctl unmask gptvswiki

sudo cp "/home/kampff/NoBlackBoxes/repos/OtherBlackBoxes/boxes/demos/gptvswiki/daemon/chromium.timer" "/etc/systemd/system/chromium.timer"
sudo cp "/home/kampff/NoBlackBoxes/repos/OtherBlackBoxes/boxes/demos/gptvswiki/daemon/gptvswiki.timer" "/etc/systemd/system/gptvswiki.timer"

sudo cp "/home/kampff/NoBlackBoxes/repos/OtherBlackBoxes/boxes/demos/gptvswiki/daemon/chromium.service" "/etc/systemd/system/chromium.service"
sudo cp "/home/kampff/NoBlackBoxes/repos/OtherBlackBoxes/boxes/demos/gptvswiki/daemon/gptvswiki.service" "/etc/systemd/system/gptvswiki.service"

sudo systemctl enable chromium.timer
sudo systemctl enable gptvswiki.timer

# Finish
echo "..."
echo "FIN"
exit 0
#FIN
