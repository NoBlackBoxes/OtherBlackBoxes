# -*- coding: utf-8 -*-
"""
Email: Template Class

@author: kampff
"""

# Import libraries
import os
import re
import Email.utilities as Utilities

# Template Class
class Template:
    def __init__(self, template_path):
        self.subject = None
        self.salutation = None
        self.body = []
        self.close = None
        self.senders = None
        self.attachments = []
        self.fields = []
        self.parse_template(template_path)
        return

    def parse_template(self, template_path):
        # Read Template
        with open(template_path, encoding='utf8') as f:
            template_lines = f.readlines()
        self.subject = template_lines[0][:-1]
        self.salutation = template_lines[1][:-1]
        body_start = 3
        body_stop = 4
        while template_lines[body_stop] != '---\n':
            body_stop += 1
        for i in range(body_start, body_stop):
            self.body.append(template_lines[i])
        self.close = template_lines[body_stop+1][:-1]
        self.senders = template_lines[body_stop+2][:-1]
        self.attachments = template_lines[body_stop+3][:-1].split(',')
        # Extract Fields
        self.parse_fields(self.subject)
        for line in self.body:
            self.parse_fields(line)
        self.fields = set(self.fields)
        return

    def parse_fields(self, line):
        fields = re.findall(r'<(.+?)>', line)
        for field in fields:
            self.fields.append(field)
        return

# FIN