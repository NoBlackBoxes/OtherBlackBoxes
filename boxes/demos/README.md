# boxes : demos

Various demonstrations of work in this repo

## Running "something" automatically after reboot (on Linux with systemd)

- Create a service/daemon ("daemon_something.service")
- Create a script to start your programme that is run by the service ("something.sh"))
- Create a timer ("something.timer") that decides when it runs

Enable services and timers
```bash
sudo systemctl unmask daemon_something
sudo cp "daemon_something.timer" "/etc/systemd/system/daemon_something.timer"
sudo cp "daemon_something.service" "/etc/systemd/system/daemon_something.service"
sudo systemctl enable daemon_something.timer
```

Example service (daemon)
```txt
[Unit]
Description=Something Daemon (service)

[Service]
Type=oneshot
User=kampff
Group=vk
ExecStart=/home/kampff/something.sh
```

Example run script
```bash
#!/bin/bash

# Cajal daemon script

# Configure CPU govenor
sudo sh -c "echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"

# Check if something client is running
if pidof "something" > /dev/null
then
  echo "Something is running."
else
  echo "Something is not running...starting"
  /home/kampff/something
fi

# Finish
exit 0
#FIN
```

Example timer
```txt
[Unit]
Description=Something Daemon (timer)

[Timer]
OnBootSec=1sec
OnUnitActiveSec=10sec

[Install]
WantedBy=timers.target
```