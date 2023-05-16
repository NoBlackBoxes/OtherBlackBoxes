#!/usr/bin/python
import os
import sys
import logging
logging.basicConfig(stream=sys.stderr)
sys.path.insert(0,"/var/www/llm/")
sys.path.insert(0,"/var/www/llm/newsfinder")
from newsfinder import app as application
application.secret_key = os.getenv("SECRET_KEY")