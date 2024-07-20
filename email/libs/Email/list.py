# -*- coding: utf-8 -*-
"""
Email: List Class

@author: kampff
"""

# Import libraries
import os
import re
import pandas as pd
import Email.utilities as Utilities

# List Class
class List:
    def __init__(self, list_path, groupby=None):
        self.groups = None
        self.parse_list(list_path, groupby)
        self.num_groups = len(self.groups)
        return

    def parse_list(self, list_path, groupby):
        frame = pd.read_excel(list_path, engine="odf")
        self.groups = frame.groupby(groupby, as_index=True)
        return

# FIN