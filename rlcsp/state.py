#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 10:41:17 2020

@author: Elena Zamaraeva
"""

import numpy as np
import json

class State:

    def __init__(self, energy, struct=[], old_energy=0, unique=True):
        self.energy = energy
        self.struct = struct
        self.old_energy = old_energy
        self.isunique = unique

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def sl(self):
        return len(self.struct)

    def struct_to_str(self):

        if type(self.struct) is np.ndarray:
            struct = self.struct.tolist()
        else:
            struct = self.struct

        return json.dumps(struct)

    def copy(self):

        return State(self.energy, self.struct.copy(), old_energy=self.old_energy, unique=self.isunique)