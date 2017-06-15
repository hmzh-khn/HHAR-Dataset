import torch
import pickle
from action_rnn import *

PICKLE_FILE = 'acc_'+PICKLE_FILE
# PICKLE_FILE = 'acc_run5_model.p'

feature_size = 3
hidden_size = 20
num_layers = 3

batch_size = 30
step_size = 10

TRAIN_USERS = TRAIN_USERS
TEST_USERS = TEST_USERS
PATH_PA = './activity_recognition'

# num_classes = len(config.ACTIONS)
num_classes = num_classes # from action_rnn
train_devs = train_devs
train_acts = train_acts


rnn, lin, train_errors, test_errors, accs = pickle.load(open(PICKLE_FILE, "rb"))
print('Accuracies', accs)



X_train, y_train, lens_train = getTrainingSequences(TRAIN_USERS, step_size, actions=train_acts, devices=train_devs, path=PATH_PA)
X_test, y_test, lens_test = getTestingSequences(TEST_USERS, step_size, actions=train_acts, devices=train_devs, path=PATH_PA)

loss1, accuracies1 = evaluate(rnn, lin, X_train, y_train, lens_train, loss_fn=nn.CrossEntropyLoss(size_average=True))
loss2, accuracies2 = evaluate(rnn, lin, X_test, y_test, lens_test, loss_fn=nn.CrossEntropyLoss(size_average=True))

accs1 = torch.sum(accuracies1, 0).t()
accs2 = torch.sum(accuracies2, 0).t()
# print('accs', accs[0][0], accs[1][0])
print("Train loss:", "%.5f" % loss1.data[0])
print("MIMO Accuracies:", "%.5f" % (float(accs1[0][0])/float(accs1[1][0])))
print("Test loss:", "%.5f" % loss2.data[0])
print("MIMO Accuracies:", "%.5f" % (float(accs2[0][0])/float(accs2[1][0])))

#     if torch.cuda.is_available():
#         X = X.cuda()
#         y = y.cuda()
#         lens = lens.cuda()

# X = X.float()
# y = y.float()
# lens = lens.float()

# # Set initial states
# h0 = Variable(torch.zeros(num_layers, X.size(0), hidden_size), requires_grad=False)
# c0 = Variable(torch.zeros(num_layers, X.size(0), hidden_size), requires_grad=False)

# if torch.cuda.is_available():
#     h0 = h0.cuda()
#     c0 = c0.cuda()

