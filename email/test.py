# -*- coding: utf-8 -*-
"""
Test bulk email sending

@author: kampff
"""
#----------------------------------------------------------
# Load environment file and variables
import os
from dotenv import load_dotenv
load_dotenv(".env")
libs_path = os.getenv('LIBS_PATH')
base_path = os.getenv('BASE_PATH')
sender = os.getenv('PROTONMAIL_USERNAME')
password = os.getenv('PROTONMAIL_SMTP_TOKEN')

# Set library paths
import sys
sys.path.append(libs_path)
#----------------------------------------------------------

# Import libraries
import os
import time
import random

# Import modules
import Email.template as Template
import Email.list as List
import Email.message as Message

# Reload libraies and modules
import importlib
importlib.reload(Template)
importlib.reload(List)
importlib.reload(Message)

#----------------------------------------------------------
# Debug
debug = False

# Specify paths
template_path = base_path + "/template.md"
list_path = base_path + "/list.ods"

# Load template
template = Template.Template(template_path)

# Load list
list = List.List(list_path, sheet="Societies", groupby=None)

# Report
num_sent = 0
for group in list.groups:
    message = Message.Message(template, group)
    message.generate()
    status = False
    if debug:
        print(message.recipients)
        print(message.subject)
        print('---')
        print(message.plain)
        print('--')
        print(message.attachments)
        print('---')
        status = True
    else:
        status = message.send(sender, password)
        delay = random.uniform(5, 15)  # Random Delay
        time.sleep(delay)        
    if status:
        num_sent += 1
print(f"{num_sent} emails sent.")
# FIN