import torch
from data_extraction import *
from torch.autograd import Variable
import math
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
import config

T = torch.Tensor

LONGEST_SEQ = 66029

"""
return numpy ndarrays sequences sorted by length for all desired data indexed by user
"""
def getSequenceSubset(users, step_size, actions=ACTIONS, devices=DEVICES, path='.'):
    seqs = []
    for u in users:
        for a in ACTIONS:
            for d in DEVICES:
                seq = Sequence(u, a, d, path=path)
                if len(seq.data) > 0:
                    examples, labels = seq.getExamples(step_size, actions=actions, devices=devices)
                    if labels.shape[0] > 0:
                        seqs.append((examples,labels))

    seqs.sort(key=lambda s: -len(s[0]))

    labels = [s[1] for s in seqs]
    seqs = [s[0] for s in seqs]
    lens = T([len(s) for s in seqs])

    padded_seqs = np.zeros((len(seqs), int(lens[0]), 3))
    for i in range(len(seqs)):
        padded_seqs[i, 0:len(seqs[i])] = seqs[i]

    padded_labels = np.zeros((len(labels), int(lens[0])))
    for i in range(len(labels)):
        padded_labels[i, 0:len(labels[i])] = labels[i][0]

    return torch.from_numpy(padded_seqs), torch.from_numpy(padded_labels), lens

def getTrainingSequences(TRAIN_USERS, step_size, actions=ACTIONS, devices=DEVICES, path='.'):
    return getSequenceSubset(TRAIN_USERS, step_size, actions=actions, devices=devices, path=path)

def getTestingSequences(TEST_USERS, step_size, actions=ACTIONS, devices=DEVICES, path='.'):
    return getSequenceSubset(TEST_USERS, step_size, actions=actions, devices=devices, path=path)

# store sequences
class Sequence:
    def __init__(self, user, action, device, path='.'):
        self.user = user
        self.action = action
        self.device = device
        self.data = read_file(action, 'pa', user=user, device=device, path=path)

    """
    Returns the accelerometer readings + labels.
    """
    def getExamples(self, skip, actions=ACTIONS, devices=DEVICES):
        out = self.data
        # print('things', devices, actions)
        out = out.loc[out['Device'].isin(devices)]
        out = out.loc[out['Action'].isin(actions)]
        # print('things', out)
        out = out.iloc[::skip, :]
        examples = out.iloc[:,3:6] # x,y,z headings
        labels = out.iloc[:,9].apply(lambda a: ACTION_KEYS[a]) # action heading, convert action labels to numbers
        return examples.as_matrix(), labels.as_matrix()

    def getUser(self):
        return self.user

    def getAction(self):
        return self.action

    def getDevice(self):
        return self.device

    def __str__(self):
        return self.data.__str__()


