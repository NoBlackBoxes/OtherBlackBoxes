# -*- coding: utf-8 -*-
"""
Build the LBB site

@author: kampff
"""
#----------------------------------------------------------
# Load environment file and variables
import os
from dotenv import load_dotenv
load_dotenv()
libs_path = os.getenv('LIBS_PATH')
base_path = os.getenv('BASE_PATH')

# Set library paths
import sys
sys.path.append(libs_path)
#----------------------------------------------------------

# Import libraries
import os
import numpy as np
import Email.utilities as Utilities

# Import modules
import Email.template as Template
import Email.list as List
import Email.message as Message

# Reload libraies and modules
import importlib
importlib.reload(Utilities)
importlib.reload(Template)
importlib.reload(List)
importlib.reload(Message)

#----------------------------------------------------------

# Specify paths
template_path = "/home/kampff/NoBlackBoxes/OtherBlackBoxes/email/template.txt"
list_path = "/home/kampff/NoBlackBoxes/OtherBlackBoxes/email/list.ods"

# Load template
template = Template.Template(template_path)

# Load list
list = List.List(list_path, groupby="Group ID")

# Report
for group in list.groups:
    message = Message.Message(template, group)
    message.render()
    print('---')

# FIN