# -*- coding: utf-8 -*-
import torch
import math
from dataclasses import dataclass
import logging as log
import numpy as np

class FuzzyArt:
    def __init__(self, pattern_size: int, vigilance: float, choice: float, learn_rate: float):
        self.pattern_size = pattern_size
        self.complemented_pattern_size = pattern_size * 2
        self.vigilance = vigilance
        self.choice = choice
        self.learn_rate = learn_rate
        self.categories = []
        self.category_counts = []
        self.category_committed = []

    def train(self, pattern: np.array):
         # check that pattern is the correct length
        if len(pattern[0]) != self.pattern_size:
            log.warn("input was the wrong size")
            return None

        # create the complemnet 
        pattern = torch.cat((pattern, 1 - pattern), axis = 1)

       # select the winner category and learn the pattern with it
        J = self.choose_category(pattern)
        self.learn_pattern(J, pattern)

        # return the category as the label
        return J

    def choose_category(self, pattern: np.array):
        N = len(self.categories)
        memberships = torch.zeros(N)
        choices = torch.zeros(N)

        # find the choice for each category
        for j in range(0, N):
            category = self.categories[j]
            fuzzy_and = np.minimum(pattern, category)
            numer = fuzzy_and.sum()
            denom = self.choice + category.sum()

            memberships[j] = numer
            choices[j] = numer / denom

        # iterate through categories by descending choice
        # until we find one that meets vigilance criteria
        # pattern_norm = pattern.sum() -> this is always pattern_size due to complementary normalization
        order = torch.argsort(choices)
        for i in range(0, len(order)):
            j = order[i]
            match = memberships[j] / self.pattern_size
            if match >= self.vigilance: return j

        # none of the categories matched
        # add a new one
        self.categories.append(torch.ones(self.complemented_pattern_size))
        self.category_counts.append(0)
        self.category_committed.append(False)
        return N

    def learn_pattern(self, J: int, pattern: np.array):
        category = self.categories[J]
        # Fast commit slow recode eq.8 in paper https://doi.org/10.1016/0893-6080(91)90056-B
        if self.category_committed[J]:
            category = self.learn_rate * torch.minimum(pattern, category) + (1 - self.learn_rate) * category
        else:
            category = torch.minimum(pattern, category)
            self.category_committed[J] = True
        self.categories[J] = category
        self.category_counts[J] += 1

def complement_code(x: torch.Tensor):
    return torch.cat((x, 1 - x), axis = 1)

if __name__ == "__main__":
    # instantiate the ART module
    art = FuzzyArt(
        pattern_size=2,
        vigilance=0.9,
        choice=0.9,
        learn_rate=0.1
    )

     # present some random patterns
    print(f"{'Label':6}: {'Pattern'} : {'Category'}")
    for i in range(0, 20):
        pattern = torch.tensor(np.array(np.random.rand(1, art.pattern_size)))  # input patter with size (1,pattern_size)
        #pattern = p1
        label = art.train(pattern)
        if label is not None:
            print(f"{label:6}: {pattern.numpy()} : {art.categories[label][0][0:art.pattern_size].numpy()}")
    print(f"Num of Cat:{len(art.categories)}")