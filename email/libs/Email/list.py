# -*- coding: utf-8 -*-
"""
Email: List Class

@author: kampff
"""

# Import libraries
import pandas as pd

# List Class
class List:
    def __init__(self, list_path, sheet=None, groupby=None):
        self.groups = None
        self.parse_list(list_path, sheet, groupby)
        self.num_groups = len(self.groups)
        return

    def parse_list(self, list_path, sheet, groupby):
        frame = pd.read_excel(list_path, sheet_name=sheet, engine="odf")
        if groupby:
            self.groups = frame.groupby(groupby, as_index=True)
        else:
            self.groups = frame.groupby("Email", as_index=True)
        return

# FIN