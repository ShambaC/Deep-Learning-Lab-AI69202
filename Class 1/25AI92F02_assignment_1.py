# %% [markdown]
# # Assignment 01: Logic Design Using MP Neurons (20 marks)

# %% [markdown]
# ## Instructions:
# 
# 1. The content that you submit must be your individual work.
# 2. Submit your code in .py as well as in .ipynb file format. Both these file submissions are required to receive credit for this assignment.
# 3. Ensure your code is well-commented and easy to follow. You can write your answers and explanations using text cells in the jupyter notebook files wherever required.
# 4. The files should be named as “(roll_number)_assignment_1”. For example, if your roll number is 23AI91R01, the code file names will be 23AI91R01_assignment_1.py and 23AI91R01_assignment_1.ipynb. You should place all these files within a single .zip file (do not upload a .rar file) and upload it to Moodle as 23AI91R01_assignment_1.zip. The zip file should only contain the .py and .ipynb files, and nothing else.
# 5. All submissions must be made through Moodle before the deadline. The submission portal will close at the specified time, and submissions via email would not be accepted.
# 6. The .ipynb file acts as your assignment report in addition to the implementation. Therefore, ensure that the .ipynb file is clear and easy to assess. To discourage plagiarism, the .py file is used to check for plagiarism with very strict deduction criteria. Anyone trying to bypass the plagiarism check with means such as gibberish text inside the code will also experience harsh deduction.
# 7. The primary TA for assignment 1 is Raj Krishan Ghosh (rajkrishanghosh@kgpian.iitkgp.ac.in). In case you have any query regarding the assignment, you can email the TA. Please do not call.

# %% [markdown]
# ## Question 1 (10 marks)
# 
# Design a half adder circuit using the minimum possible number of MP neurons.

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from typing import List

# %%
# MP Neuron Model with Inhibitory Input Support

def mp_neuron(inputs: np.ndarray, threshold: int, inhibitory_indices: List[int]=[]) -> int:
  """
  Method to simulate an MP neuron

  Args
  ----
    inputs : np.ndarray
        Array of binary inputs (0s and 1s)
    threshold : int
        The threshold value for firing
    inhibitory_indices : List[int]
        List of indices in the inputs array that are inhibitory

  Returns
  -------
    int
        1 if neuron fires, 0 otherwise
  """
  net_input = np.sum(inputs)
  for idx in inhibitory_indices:
    if inputs[idx] >= 0.5:
      return 0
  return 1 if net_input >= threshold else 0

# %%
# Boolean Functions
## AND Function
def and_function(x1: int, x2: int) -> int:
  return mp_neuron(np.array([x1, x2]), 2)

## OR Function
def or_function(x1: int, x2: int) -> int:
  return mp_neuron(np.array([x1, x2]), 1)

## NOT Function using inhibitory input
def not_function(x1: int) -> int:
  return mp_neuron(np.array([x1]), -np.inf, inhibitory_indices=[0])

## XOR method
def xor_function(x1: int, x2: int) -> int:
  or_output = or_function(x1, x2)
  and_output = and_function(x1, x2)

  # XOR is basically the OR output inhibited by the AND output
  # When x1 = 1 and x2 = 1, OR output is 1 and AND output is 1, so it inhibits the OR output with a final output of 0
  # When x1 = 1 and x2 = 0 or x1 = 0 and x2 = 1, OR output is 1 and AND output is 0, so no inhibition occurs and final output is 1
  # When x1 = 0 and x2 = 0, OR output is 0 and AND output is 0, so no inhibition occurs and final output is 0
  xor_output = mp_neuron(np.array([or_output, and_output]), 1, inhibitory_indices=[1])
  return xor_output

# %%
def half_adder(x1: int, x2: int) -> tuple[int, int]:
  """
  Half Adder using XOR and AND MP neurons
  The sum uses a total of 3 MP neurons, 1 for OR, 1 for AND and 1 for the main XOR logic.
  The carry uses a single AND MP neuron.
  """
  sum = xor_function(x1, x2)
  carry = and_function(x1, x2)
  return sum, carry

# %% [markdown]
# ## Question 2 (2 marks)
# 
# Test the half adder that you designed on all possible input combinations.

# %%
inputs_ADD = [[0, 0], [0, 1], [1, 0], [1, 1]]
ground_truth_ADD = [0, 1, 1, 0]
ground_truth_CARRY = [0, 0, 0, 1]

outputs_ADD = []
outputs_CARRY = []
for i in range(len(inputs_ADD)):
  input = inputs_ADD[i]
  sum, carry = half_adder(input[0], input[1])
  outputs_ADD.append(sum)
  outputs_CARRY.append(carry)

print("Inputs for ADD")
print(inputs_ADD)
print("Ground Truth for ADD")
print(ground_truth_ADD)
print("Outputs for ADD")
print(outputs_ADD)
print("Ground Truth for CARRY")
print(ground_truth_CARRY)
print("Outputs for CARRY")
print(outputs_CARRY)

# %% [markdown]
# ## Question 3 (6 marks)
# 
# Using two half adders and an OR gate, design a full adder.

# %% [markdown]
# ### Full adder with two half adders and OR
# A full adder has 3 bit inputs (two main inputs and one carry in) and produces sum and carry as output.
# 
# Sum = A XOR B XOR C (A, B, C are the inputs)
# 
# This expression can be written as (A XOR B) XOR C. One half adder can XOR A and B while the second half adder can XOR the result with C.
# 
# The carry output of these two half adders are OR'd to get the final carry output

# %%
# Full adder
def full_adder(x1: int, x2: int, c_in: int) -> tuple[int, int]:
  """
  Full Adder using two Half Adders and an OR MP neuron
  """
  sum_1, carry_1 = half_adder(x1, x2)
  final_sum, carry_2 = half_adder(sum_1, c_in)
  final_carry = or_function(carry_1, carry_2)
  return final_sum, final_carry

# %% [markdown]
# ## Question 4 (2 marks)
# 
# Test the full adder that you designed on all the possible input combinations.

# %%
inputs_ADD = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
ground_truth_ADD = [0, 1, 1, 0, 1, 0, 0, 1]
ground_truth_CARRY = [0, 0, 0, 1, 0, 1, 1, 1]

outputs_ADD = []
outputs_CARRY = []
for i in range(len(inputs_ADD)):
  input = inputs_ADD[i]
  sum, carry = full_adder(input[0], input[1], input[2])
  outputs_ADD.append(sum)
  outputs_CARRY.append(carry)

print("Inputs for ADD")
print(inputs_ADD)
print("Ground Truth for ADD")
print(ground_truth_ADD)
print("Outputs for ADD")
print(outputs_ADD)
print("Ground Truth for CARRY")
print(ground_truth_CARRY)
print("Outputs for CARRY")
print(outputs_CARRY)


