# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 19:21:47 2020

@author: Zheng Wen
"""

class Combination:
    def combine(self, n, k):
        # write your code here
        self.res = []
        if n == 0 or k == 0:
            return []
        self.dfs([], 1, 0, n, k)
        return self.res
    
    def dfs(self, combination, start, length, n, k):
        if length == k:
            self.res.append(sorted(combination[:]))
            return
        
        for i in range(start, n + 1):
            if i in combination:
                return
            combination.append(i)
            self.dfs(combination, start + 1, length + 1, n, k)
            combination.pop()