# Email

Instructions to generate and send automated emails

## Requirements

```bash
pip install numpy pandas python-dotenv odfpy
```

## Recipient (Group) List
Create an ODS file with the following format with columns with field names, must have "First Name" and "Email"

## Sending
- You will need a password manager installed

```bash
sudo pacman -S pass

# Generate key for pass manager
gpg --generate-key # do not enter a passphrase
pass init '{your gpg key id}'
```
- Install the Proton Mail Bridge (binary is in AUR)
- Run...login and sync

```bash
protonmail-bridge --cli
>>> login
```
- Note "bridge" password (needed in Python scripts)

```bash
>>> info
```

- Change smtp-security to use SSL

```bash
>>> change smtp-security
>>> exit
```

***Important***: Copy password 

***Just this next step to run later...***

- Start proton mail bridge
```bash
nohup protonmail-bridge --noninteractive > bridge_log.txt 2>&1 &
disown
```

- Send a test email

```python
import smtplib
from email.mime.text import MIMEText
sender = 'kampff@voight-kampff.tech'
receiver = 'adam.kampff@gmail.com'
message = MIMEText("Hello, world!")
message['Subject'] = "Message subject!"
message['From'] = sender
message['To'] = receiver
smtp = smtplib.SMTP_SSL('127.0.0.1', 1025) # 1025 - port from proton mail bridge
smtp.login(sender, 'password') # email and password
smtp.sendmail(sender, [receiver], message.as_string())
```