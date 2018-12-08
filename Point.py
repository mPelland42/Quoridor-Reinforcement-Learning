# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 15:36:55 2018

@author: Sam
"""

class Point:
    
    X = None
    Y = None
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        
    def addToX(self, dx):
        self.X += dx
        
    def addToY(self, dy):
        self.Y += dy
        
    def xstr(self,s):
        if s is None:
            return 'NULL'
        return str(s)

    def __str__(self):
        return "(" + self.xstr(self.X) + "," + self.xstr(self.Y) + ")"
    
    def toTuple(self):
        return (self.X, self.Y)
    def __eq__(self, other):
        #try catch because sometimes comparing actions compars a point to None
        try:
            return self.X == other.X and self.Y == other.Y
        except:
            return False
    
    #needed for A*
    def __hash__(self):
        return hash(self.toTuple())
    
    def __add__(self, other):
        return Point(self.X + other.X, self.Y + other.Y)
    
    #needed for A*
    def __lt__(self, other):
        return self.X < other.X or (self.X == other.X and self.Y < other.Y)
    
    def __repr__(self):
        return  "Point (" + str(self.X) + ", " + str(self.Y) + ")"