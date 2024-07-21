# -*- coding: utf-8 -*-
"""
Email: Template Class

@author: kampff
"""

# Import libraries
import re

# Template Class
class Template:
    def __init__(self, template_path):
        self.subject = None
        self.salutation = None
        self.text = []
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
        text_start = 3
        text_stop = 4
        while template_lines[text_stop] != '---\n':
            text_stop += 1
        for i in range(text_start, text_stop):
            self.text.append(template_lines[i])
        self.close = template_lines[text_stop+1][:-1]
        self.senders = template_lines[text_stop+2][:-1]
        self.attachments = template_lines[text_stop+3][:-1].split(',')
        # Extract Fields
        self.parse_fields(self.subject)
        for line in self.text:
            self.parse_fields(line)
        for line in self.attachments:
           self.parse_fields(line)
        self.fields = set(self.fields)
        return

    def parse_fields(self, line):
        fields = re.findall(r'<(.+?)>', line)
        for field in fields:
            self.fields.append(field)
        return

# FIN