# -*- coding: utf-8 -*-
"""
Email: Message Class

@author: kampff
"""

# Import libraries
import os
import re
import pandas as pd
import Email.utilities as Utilities

# Message Class
class Message:
    def __init__(self, template, group):
        self.template = template
        self.group = group
        self.num_recipents = len(group[1])
        return

    def render(self):
        subject = self.replace_fields(self.template.fields, self.template.subject)
        print(f"Subject: {subject}\n")
        salutation = self.generate_salutation(self.group)
        print(f"{salutation}\n")
        for line in self.template.body:
            line = self.replace_fields(self.template.fields, line)
            print(f"{line}")
        print(self.template.close)
        print(self.template.senders)
        return

    def replace_fields(self, fields, line):
        for field in fields:
            if field == 'Address':
                value = "Adress Formated Correctly\n12 Street"                 
            else:    
                value = self.group[1][field].values[0]
            line = line.replace(f"<{field}>", str(value))        
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

# FIN