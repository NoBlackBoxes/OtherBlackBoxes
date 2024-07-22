# -*- coding: utf-8 -*-
"""
Email: Message Class

@author: kampff
"""

# Import libraries
import numpy as np
import pandas as pd
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# Message Class
class Message:
    def __init__(self, template, group):
        self.template = template
        self.group = group
        self.num_recipents = len(group[1])
        self.recipients = []
        self.subject = None
        self.body = None
        self.attachments = []
        return

    def generate(self):
        self.recipients = self.group[1]['Email'].values
        self.subject = self.replace_fields(self.template.fields, self.template.subject)
        body = []
        salutation = self.generate_salutation(self.group)
        body.append(f"{salutation}\n\n")
        for line in self.template.text:
            line = self.replace_fields(self.template.fields, line)
            if line is not None:
                body.append(f"{line}")
        body.append("\n")
        body.append(self.template.close + "\n")
        body.append(self.template.senders + "\n")
        self.body = "".join(body)
        attachments = []
        for attachment in self.template.attachments:
            attachments.append(self.replace_fields(self.template.fields, attachment))
        self.attachments = attachments
        return

    def replace_fields(self, fields, line):
        for field in fields:
            value = self.group[1][field].values[0]
            line = line.replace(f"{{{field}}}", str(value))
        # If line has "empty" field (nan) and whitespace, return None
        trimmed = line.strip()
        if trimmed == "nan":
            line = None
        return line

    def generate_salutation(self, group):
        names = group[1]['First Name'].values
        salutation = []
        salutation.append(f"{self.template.salutation} ")
        if self.num_recipents > 2:
            for name in names[:-1]:
                salutation.append(f"{name}, ")
            salutation.append(f"and {names[-1]},")
        elif self.num_recipents == 2:
            salutation.append(f"{names[0]} and {names[1]},")
        else:
            salutation.append(f"{names[0]},")
        salutation = "".join(salutation)
        return salutation
    
    def send(self, sender, password):
        email = MIMEMultipart()
        email['From'] = f"NoBlackBoxes <{sender}>"
        email['To'] = ", ".join(self.recipients)
        email['Subject'] = self.subject
        email.attach(MIMEText(self.body, 'plain'))
        # !! Atach Attachments !!
        smtp = smtplib.SMTP('smtp.protonmail.ch', 587)
        smtp.ehlo()  # send the extended hello to our server
        smtp.starttls()  # tell server we want to communicate with TLS encryption
        smtp.login(sender, password)
        smtp.sendmail(sender, self.recipients, email.as_string())
        print(f"Sent ({self.subject})")
        return

# FIN