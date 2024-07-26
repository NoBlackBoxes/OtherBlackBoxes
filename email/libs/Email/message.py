# -*- coding: utf-8 -*-
"""
Email: Message Class

@author: kampff
"""

# Import libraries
import os
import numpy as np
import pandas as pd
import markdown
import smtplib
from email.message import EmailMessage

# Message Class
class Message:
    def __init__(self, template, group):
        self.template = template
        self.group = group
        self.num_recipents = len(group[1])
        self.recipients = []
        self.subject = None
        self.plain = None
        self.html = None
        self.attachments = []
        return

    def generate(self):
        self.recipients = self.group[1]['Email'].values
        self.subject = self.replace_fields(self.template.fields, self.template.subject)
        body = []
        for line in self.template.body:
            line = self.replace_fields(self.template.fields, line)
            if line is not None:
                body.append(f"{line}")
        body.append("\n")
        self.plain = "".join(body)
        self.html = "<html>\n<body>\n" + markdown.markdown(self.plain) + "\n</html>\n</body>\n"
        attachments = []
        for attachment in self.template.attachments:
            attachments.append(self.replace_fields(self.template.fields, attachment))
        self.attachments = attachments
        return

    def replace_fields(self, fields, line):
        for field in fields:
            if (len(field) > 3) and (field[-3:] == '(s)'):
                values = self.group[1][field[:-3]].values
                plural = []
                if self.num_recipents > 2:
                    for name in values[:-1]:
                        plural.append(f"{name}, ")
                    plural.append(f"and {values[-1]}")
                elif self.num_recipents == 2:
                    plural.append(f"{values[0]} and {values[1]}")
                else:
                    plural.append(f"{values[0]}")
                plural = "".join(plural)
                line = line.replace(f"{{{field}}}", str(plural))
            else:
                value = self.group[1][field].values[0]
                line = line.replace(f"{{{field}}}", str(value))
        # If line has "empty" field (nan) and whitespace, return None
        trimmed = line.strip()
        if trimmed == "nan":
            line = None
        else:
            # Remove any other "nans"
            line = line.replace(f"nan", '')
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
        email = EmailMessage()
        email['From'] = f"NoBlackBoxes <{sender}>"
        email['To'] = ", ".join(self.recipients)
        email['Subject'] = self.subject
        email.set_content(self.plain)
        email.add_alternative(self.html, subtype="html")
        if len(self.attachments) > 0:
            for attachment in self.attachments:
                attachment_suffix = attachment.split('.')[-1]
                with open(attachment, "rb") as f:
                    if attachment_suffix == 'pdf':
                        email.add_attachment(f.read(), filename=attachment, maintype="application", subtype="pdf")
                    elif attachment_suffix in ['jpg', 'jpeg', 'png']:
                        email.add_attachment(f.read(), filename=attachment, maintype="image", subtype=attachment_suffix)
                    else:
                        print(f"Unsupported Attachment Type: {attachment_suffix}")
                        exit(-1)
        smtp = smtplib.SMTP('smtp.protonmail.ch', 587)
        smtp.ehlo()  # send the extended hello to our server
        smtp.starttls()  # tell server we want to communicate with TLS encryption
        smtp.login(sender, password)
        smtp.sendmail(sender, self.recipients, email.as_string())
        print(f"Sent ({self.subject})")
        return

# FIN