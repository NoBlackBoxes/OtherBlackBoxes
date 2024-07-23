# -*- coding: utf-8 -*-
"""
Test bulk email sending

@author: kampff
"""
#----------------------------------------------------------
# Load environment file and variables
import os
from dotenv import load_dotenv
load_dotenv()
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
import numpy as np

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
debug = True

# Specify paths
template_path = base_path + "/template.md"
list_path = base_path + "/list.ods"

# Load template
template = Template.Template(template_path)
print(template.fields)

# Load list
list = List.List(list_path, groupby="Group ID")

# Report
for group in list.groups:
    message = Message.Message(template, group)
    message.generate()
    if debug:
        print(message.recipients)
        print(message.subject)
        print('---')
        print(message.plain)
        print('--')
        print(message.attachments)
        print('---')
    else:
        message.send(sender, password)
# FIN