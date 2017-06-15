"""
Author      : Hamzah Khan
Class       : HMC CS 158 Final Project
Date        : 2017 Apr 11
Description : Data visualization and extraction for the HHAR data set.
"""

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

from config import *


"""
This function extracts the action sequences and puts them into individual csv
files.
Input: fname
"""
def extract_complete_action_seqs(fname, path='./'):
  full_name = make_fname(fname, path=path)
  df = pd.read_csv(full_name+'.csv')

  act_dfs = {}

  # for action in ACTIONS:
  #   act_dfs[action] = []

  for user in USERS:
    usr_df = df[df.User == user]
    for action in ACTIONS:
      print("identifying samples for action", action, "for user", user)
      for device in DEVICES:
        act_df = usr_df[usr_df.Action == action]
        dev_act = act_df[act_df.Device==device]
        dev_act.sort_values(by='Index', ascending=True)
        full_name = make_fname(fname, action=action, path=path, user=user, device=device)
        dev_act.to_csv(full_name+'.csv', index=False)
      print("saved", action, "data for user", user, "to file")

      # act_dfs[action].append(X_act)

  # for action in ACTIONS:
  #   full_name = make_fname(fname, action=action, path=path)
  #   act_df = pd.concat(act_dfs[action])
  #   act_df.to_csv(full_name+'.csv', index=False)
  #   print("saved", action, "data for all users to file")

"""
Construct filename as necessary.
"""
def make_fname(fname, path='./', user='', device='', action=None):
  if fname in RAW_DATA_FILENAMES:
    if action:
      return path+'/'+user+device+fname+'_'+action
    else:
      return path+'/'+fname
  else:
    if action:
      return path+'/'+user+device+FILENAMES[fname]+'_'+action
    else:
      return path+'/'+FILENAMES[fname]


"""
Read csv files that contain specific actions for specific users.
"""
def read_file(action, fname, user='', device='', path='./'):
  if user:
    full_name = make_fname(fname, action=action, user=user, path=path, device=device)
    return pd.read_csv(full_name+'.csv')
  else:
    ppl_data = []
    for user in USERS:
      full_name = make_fname(fname, action=action, user=user, path=path, device=device)
      ppl_data.append(full_name)
    # returns action for all users
    return pd.concat([pd.read_csv(f+'.csv') for f in ppl_data])

"""
This function extracts the start and end indices of the actions sequentially by 
action type.
"""
def extract_bounds(fname, path='./'):
  action_bounds = {}

  for action in ACTIONS:
    action_bounds[action] = []
    df = read_file_action(action, fname, path=path)
    start = df.iloc[0]['Index']
    prev = df.iloc[0]['Index']
    next = df.iloc[0]['Index']
    for index, row in df.iterrows():
      next = row['Index']
      print(index, prev, next, next-prev)


      # if more than 1 index later, 
      # then create a bound because it is an action change
      if next - prev > 1:
        # print((start, prev, next), next - prev)
        action_bounds[action].append((start, prev))
        start = next

      prev = next

  return action_bounds

# count number of examples in each set. Note that 
def countNumExamples():
  counts = {}
  max = 0
  for user in USERS:
    counts[user] = {}
    for action in ACTIONS:
      counts[user][action] = {}
      for device in DEVICES:
        counts[user][action][device] = {}
        contents = read_file(action, 'pa', user, device, path='./activity_recognition')
        counts[user][action][device] = len(contents)
        # print(action, user, device, len(contents))
        if counts[user][action][device] == 0:
          print(user, action, device, 0)
        if max < counts[user][action][device]:
          max = counts[user][action][device]
  return counts, max

