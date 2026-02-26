import json
from typing import List, Dict
import pdb

SYSTEM_BASE_INTRO = (
    "You are athe best programmer in the world.\n"
    "You will be asked to determine the time complexity of the following code, but you are a part of a panel of experts and must vote for the most accurate time complexity.\n"
    "As a reminder here is a guide to the different possible complexity classes you can vote for:\n"
    "{expertise_guide}\n"
    "{voting_format_guide}\n"
    "Do not hesitate to use any other supplementary materials you need for the task.\n\n"
)

SIMPLE_VOTE_PROMPT = (
    "To submit your vote for the correct time complexity, you will have to submit a JSON object with the following format:\n"
    "{\n"
    ' "complexity": "the time complexity class you think is the most accurate"\n'
    "}\n\n"
    "Your choices for the time complexity class are: constant, logn, linear, nlogn, quadratic, cubic, and exponential.\n"
    "Make sure to choose the one you think is the most accurate.\n"
)

MAJORITY_VOTE_PROMPT = (
    "To submit your vote for the correct time complexity, you will have to submit a JSON object with the following format:\n"
    "{\n"
    ' "constant": 1 for "yes" and 0 for "no",\n'
    ' "logn": 1 for "yes" and 0 for "no",\n'
    ' "linear": 1 for "yes" and 0 for "no",\n'
    ' "nlogn": 1 for "yes" and 0 for "no",\n'
    ' "quadratic": 1 for "yes" and 0 for "no",\n'
    ' "cubic": 1 for "yes" and 0 for "no",\n'
    ' "exponential": 1 for "yes" and 0 for "no"\n'
    "}\n\n"
    "You will only be allowed to vote for one time complexity class, so make sure to choose the one you think is the most accurate.\n"
)

RANKED_CHOICE_PROMPT = (
    "To submit your vote for the correct time complexity, you will have to submit a JSON object with the following format:\n"
    "{\n"
    ' "constant": 1 for "most likely", 2 for "second most likely", 3 for "third most likely", 4 for "fourth most likely", 5 for "fifth most likely", 6 for "least likely", and 0 for "not likely",\n'
    ' "logn": 1 for "most likely", 2 for "second most likely", 3 for "third most likely", 4 for "fourth most likely", 5 for "fifth most likely", 6 for "least likely", and 0 for "not likely",\n'
    ' "linear": 1 for "most likely", 2 for "second most likely", 3 for "third most likely", 4 for "fourth most likely", 5 for "fifth most likely", 6 for "least likely", and 0 for "not likely",\n'
    ' "nlogn": 1 for "most likely", 2 for "second most likely", 3 for "third most likely", 4 for "fourth most likely", 5 for "fifth most likely", 6 for "least likely", and 0 for "not likely",\n'
    ' "quadratic": 1 for "most likely", 2 for "second most likely", 3 for "third most likely", 4 for "fourth most likely", 5 for "fifth most likely", 6 for "least likely", and 0 for "not likely",\n'
    ' "cubic": 1 for "most likely", 2 for "second most likely", 3 for "third most likely", 4 for "fourth most likely", 5 for "fifth most likely", 6 for "least likely", and 0 for "not likely",\n'
    ' "exponential": 1 for "most likely", 2 for "second most likely", 3 for "third most likely", 4 for "fourth most likely", 5 for "fifth most likely", 6 for "least likely", and 0 for "not likely"\n'
    "}\n\n"
    "You will have to rank all the time complexity classes from most likely to least likely, so make sure to choose the one you think is the most accurate as your most likely choice.\n"
)

APPROVAL_VOTE_PROMPT = (
    "To submit your vote for the correct time complexity, you will have to submit a JSON object with the following format:\n"
    "{\n"
    ' "constant": 1 for "approve" and 0 for "disapprove",\n'
    ' "logn": 1 for "approve" and 0 for "disapprove",\n'
    ' "linear": 1 for "approve" and 0 for "disapprove",\n'
    ' "nlogn": 1 for "approve" and 0 for "disapprove",\n'
    ' "quadratic": 1 for "approve" and 0 for "disapprove",\n'
    ' "cubic": 1 for "approve" and 0 for "disapprove",\n'
    ' "exponential": 1 for "approve" and 0 for "disapprove"\n'  
    "}\n\n"
    "You will only be allowed to approve multiple time complexity classes at the same time, so if you think multiple time complexity classes are accurate, make sure to approve all of them and disapprove the ones you think are not accurate.\n"
)

EXPERTISE_GUIDE = (
    'constant:'
    "\tConstant time complexity means that the execution time of a function does not depend on the size of the input.\n"
    "\tRegardless of how large the input is, the function completes in a fixed number of operations.\n"
    'logn:'
    "\tLogarithmic complexity means that the number of operations grows proportionally to the logarithm of the input size.\n"
    "\tThis often occurs in divide-and-conquer algorithms or binary search-like structures.\n\n"
    "\t## Logical Steps to Determine logarithmic time complexity:\n"
    "\t1. Identify if the input size is being reduced by a constant factor (e.g., half) at each iteration.\n"
    "\t2. Look for algorithms that involve binary search, tree traversal (balanced trees), or divide-and-conquer.\n"
    "\t3. Ensure the number of operations does not scale linearly but instead decreases exponentially.\n"
    "\t4. If the loop or recursion reduces the problem size logarithmically, classify it as the logarithmic complexity.\n"
    'linear:'
    "\tLinear complexity means that the execution time grows proportionally with the input size.\n"
    "\tThis is typical in single-loop iterations over an array or list.\n"
    'nlogn:'
    "\tO(n log n) complexity is common in efficient sorting algorithms like Merge Sort or Quick Sort.\n"
    "\tIt arises when a problem is divided into smaller subproblems while still iterating over the input.\n\n"
    "\t## Logical Steps to Determine nlogn time complexity:\n"
    "\t1. Identify if the input is being divided into smaller parts recursively (logarithmic factor).\n"
    "\t2. Ensure that a linear operation is performed at each level of recursion.\n"
    "\t3. Look for sorting algorithms like Merge Sort, Quick Sort (average case), or efficient divide-and-conquer solutions.\n"
    "\t4. If the algorithm involves dividing the problem and processing each part linearly, classify it as nlogn time complexity.\n"
    'quadratic:'
    "\tQuadratic complexity occurs when an algorithm has double nested loops, where each loop iteration depends on the input size.\n"
    'cubic:'
    "\tCubic complexity occurs when an algorithm has three nested loops iterating over the input size.\n\n"
    "\t## Logical Steps to Determine cubic time complexity:\n"
    "\t1. Identify if there are three nested loops iterating from 0 to n.\n"
    "\t2. Ensure that each element is compared or processed against every pair of elements.\n"
    "\t3. Look for brute-force solutions that check all triplets in an input set.\n"
    "\t4. If the number of operations scales as the cube of the input size, classify it as cubic complexity.\n"
    'exponential:'
    "\tExponential complexity occurs when the number of operations doubles with each additional input element.\n"
    "\tThis is common in brute-force recursive algorithms, such as solving the Traveling Salesman Problem.\n\n"
    "\t## Logical Steps to Determine exponential time complexity:\n"
    "\t1. Identify if the function calls itself recursively, doubling the number of calls at each step.\n"
    "\t2. Look for recursion that does not significantly reduce the input size in each step.\n"
    "\t3. Check for exhaustive searches, backtracking algorithms, or recursive Fibonacci calculations.\n"
    "\t4. If the number of operations grows exponentially with input size, classify it as exponential complexity.\n"
)

def build_single_majority_vote(src: str, **_) -> List[Dict[str, str]]:
    return SYSTEM_BASE_INTRO.format(expertise_guide=EXPERTISE_GUIDE, voting_format_guide=MAJORITY_VOTE_PROMPT) + src

def build_single_ranked_choice(src: str, **_) -> List[Dict[str, str]]:
    return SYSTEM_BASE_INTRO.format(expertise_guide=EXPERTISE_GUIDE, voting_format_guide=RANKED_CHOICE_PROMPT) + src

def build_single_approval_vote(src: str, **_) -> List[Dict[str, str]]:
    return SYSTEM_BASE_INTRO.format(expertise_guide=EXPERTISE_GUIDE, voting_format_guide=APPROVAL_VOTE_PROMPT) + src

def build_single_simple(src: str, **_) -> List[Dict[str, str]]:
    return SYSTEM_BASE_INTRO.format(expertise_guide=EXPERTISE_GUIDE, voting_format_guide=SIMPLE_VOTE_PROMPT) + src