import torch
import torch.nn as nn
from torch.autograd import Variable
from data_extraction import *
from model import *
import math
from torch.nn.utils.rnn import pad_packed_sequence
import config
import pickle

TRAIN_USERS = ['a', 'b', 'c', 'e', 'g', 'h', 'i']
TEST_USERS = ['d', 'f']
PATH_PA = './activity_recognition'

PICKLE_FILE = "run9_model.p"

feature_size = 3
hidden_size = 20
num_layers = 2

batch_size = 12
step_size = 10

num_classes = 2 #len(config.ACTIONS)
num_epochs = 1000

train_devs = ['samsungold_1','samsungold_2']
train_acts = ['bike', 'sit']

is_lstm = True
is_LOOCV = True

def train(rnn, lin, X, y, lens, percent_train=0.7, num_epochs=10, loss_fn=nn.CrossEntropyLoss(size_average=True), learning_rate=0.05, optimizer=None):
    train_errors = torch.zeros(num_epochs)
    test_errors = torch.zeros(num_epochs)

    X = X.float()
    y = y.float()
    lens = lens.float()

    # split into sets for training (70%), cross_validation (30%) - outputs np ndarrays
    X_train, y_train, lens_train, X_test, y_test, lens_test = training_test_split(X, y, lens, percent_train=percent_train)

    # Make sequence, labels for cv set
    test_seq, test_labels = make_cv_sequence(X_test, y_test, lens_test)

    # Set initial states
    h0 = Variable(torch.zeros(num_layers, batch_size, hidden_size), requires_grad=True)
    c0 = Variable(torch.zeros(num_layers, batch_size, hidden_size), requires_grad=True)
    # print('sizes', X_test.size(), y_test.size())
    h0_test = Variable(torch.zeros(num_layers, X_test.size(0), hidden_size), requires_grad=True)
    c0_test = Variable(torch.zeros(num_layers, X_test.size(0), hidden_size), requires_grad=True)

    if torch.cuda.is_available():
        h0 = h0.cuda()
        c0 = c0.cuda()
        h0_test = h0_test.cuda()
        c0_test = c0_test.cuda()

    # run a loop for each epoch of training
    for t in range(num_epochs):
        # create training batches (padded sequences) of size batch_size
        batch, labels = make_training_batch(X_train, y_train, lens_train, batch_size)

        y_pred = None
        if is_lstm:
            y_pred, _ = rnn(batch, (h0, c0))
        else:
            y_pred, _ = rnn(batch, h0)

        var1 = lin(y_pred.data)
        var2 = labels.data

        # print('var sizes', var1.data.size(), var2.data.size(), y)
        loss = loss_fn(var1, var2)

        # calculate test errors
        test_pred = None
        if is_lstm:
            test_pred, _ = rnn(test_seq, (h0_test, c0_test))
        else:
            test_pred, _ = rnn(test_seq, h0_test)
        

        test_var1 = lin(test_pred.data)
        test_var2 = test_labels.data
        test_loss = loss_fn(test_var1, test_var2)

        # record training and test errors
        train_errors[t] = loss.data[0]
        test_errors[t] = test_loss.data[0]

        str1 = "%.5f" % train_errors[t]
        str2 = "%.5f" % test_loss.data[0]
        print("{}\t steps: training loss - {}\t, testing loss - {}\t".format(t, str1, str2))
        

        # Zero the gradients before running the backward pass.
        lin.zero_grad()
        rnn.zero_grad()

        # Backward pass: compute gradient of the loss with respect to all the learnable
        # parameters of the rnn. Internally, the parameters of each Module are stored
        # in Variables with requires_grad=True, so this call will compute gradients for
        # all learnable parameters in the rnn.
        loss.backward(retain_variables=True)
        
        # Update the weights using Adam-optimized gradient descent
        optimizer.step()

    return rnn, lin, train_errors, test_errors

"""
Identify a training and cross-validation subset of the data given a np X, y 
input.
"""
def training_test_split(X, y, lens, percent_train=0.6):
    num_train_examples = math.ceil(percent_train*X.size(0))
    idx = torch.randperm(X.size(0)) # sorting maintains the order of the lengths of sequences
    if torch.cuda.is_available():
        idx = idx.cuda()

    train_idx, _ = torch.sort(idx[:num_train_examples], 0)
    test_idx, _  = torch.sort(idx[num_train_examples:], 0)

    X_train = torch.index_select(X, 0, train_idx)
    X_test  = torch.index_select(X, 0, test_idx)

    y_train = torch.index_select(y, 0, train_idx)
    y_test  = torch.index_select(y, 0, test_idx)

    lens_train = torch.index_select(lens, 0, train_idx).float()
    lens_test = torch.index_select(lens, 0, test_idx).float()

    return X_train, y_train, lens_train, X_test, y_test, lens_test


"""
Create one batch (of type packed_sequence) of the desired batch size, and annotate
it with sequence labels, lengths.
"""
def make_training_batch(X, y, lens, batch_size):
    idx = torch.randperm(X.size(0))
    if torch.cuda.is_available():
        idx = idx.cuda()

    batch_idx, _ = torch.sort(idx[:batch_size])

    seqs = torch.index_select(X, 0, batch_idx).float()
    labels = torch.index_select(y, 0, batch_idx).float()

    # convert to list for passing to padding function
    batch_lens = torch.index_select(lens, 0, batch_idx).float().tolist()

    # convert to Variables, then to batched padded sequences and train the batches
    batch = Variable(seqs, requires_grad=True)
    labels = Variable(labels.long(), requires_grad=False)

    batch = pack_padded_sequence(batch, batch_lens, batch_first=True)
    labels = pack_padded_sequence(labels, batch_lens, batch_first=True)

    return (batch, labels)


# """
# Create batches (of type packed_sequence) of the desired batch size, and annotate
# them with sequence labels, lengths.
# """
# def make_training_batches(X, y, lens, batch_size, num_batches=10):
#     batches = []
#     batch_labels = []

#     for b in range(num_batches):
#         idx = torch.randperm(X.size(0))
#         # print('idx sort', idx[:batch_size], torch.sort(idx[:batch_size]))
#         batch_idx, _ = torch.sort(idx[:batch_size])
#         # batch_idx = sorted(idx[:batch_size])

#         seqs = torch.index_select(X, 0, batch_idx).float()
#         labels = torch.index_select(y, 0, batch_idx).float()
#         # convert to list for passing to padding function
#         batch_lens = torch.index_select(lens, 0, batch_idx).float().tolist()

#         # convert to Variables, then to batched padded sequences and train the batches
#         batch = Variable(seqs, requires_grad=True)
#         labels = Variable(labels.long(), requires_grad=False)
#         # print('Batch Info', batch.size())
#         batch = pack_padded_sequence(batch, batch_lens, batch_first=True)
#         labels = pack_padded_sequence(labels, batch_lens, batch_first=True)
#         # print('Sequence dims', seqs.data.size())
#         # print('batch info')
#         batches.append(batch)
#         batch_labels.append(labels)
#     return (batches, batch_labels)


"""
For making the cv_test sequence for test error.
"""
def make_cv_sequence(X, y, lens):
    # convert to list for passing to padding function
    batch_lens = lens.tolist()

    # convert to Variables, then to batched padded sequences and train the batches
    seq = Variable(X, requires_grad=False)
    labels = Variable(y.long(), requires_grad=False)

    seq = pack_padded_sequence(seq, batch_lens, batch_first=True)
    labels = pack_padded_sequence(labels, batch_lens, batch_first=True)

    return (seq, labels)


"""
Evaluation of model on test data.
"""
def evaluate(rnn, lin, X_test, y_test, lens_test, loss_fn=nn.CrossEntropyLoss(size_average=True), show=True):
    if show:
        print('EVALUATION')
        print('----------')

    h0 = Variable(torch.zeros(num_layers, X_test.size(0), hidden_size), requires_grad=False)
    c0 = Variable(torch.zeros(num_layers, X_test.size(0), hidden_size), requires_grad=False)
    if torch.cuda.is_available():
        h0 = h0.cuda()
        c0 = c0.cuda()

    # make entire thing into packed sequence
    X_test = Variable(X_test.float(), requires_grad=False)
    if torch.cuda.is_available():
        X_test = X_test.cuda()
        y_test = y_test.cuda()
    seq = pack_padded_sequence(X_test, lens_test.float().tolist(), batch_first=True)

    y_pred = None
    if is_lstm:
        y_pred, _ = rnn(seq, (h0, c0))
    else:
        y_pred, _ = rnn(seq, h0)
    y_pred, y_pred_lens = pad_packed_sequence(y_pred, batch_first=True)

    # calculate accuracy, loss
    # numerator, total
    accuracies = torch.zeros(len(y_pred),2) # For MISO
    for p in range(y_pred.size(0)):
        pred = lin(y_pred[p])
        # print('prediction', pred.data, y_test)
        idx = torch.range(1, lens_test[p]).long() - 1
        
        if torch.cuda.is_available():
            idx = idx.cuda()

        # print('Sizes II', idx.size(), y_test[p].index_select(0,idx).size(), pred.size(), torch.max(pred, 1)[1].float().size(), torch.squeeze(torch.max(pred, 1)[1].float()).size())
        # print('sizes III', y_test[p].float().index_select(0,idx).size(), torch.squeeze(torch.max(pred, 1)[1].float()).data.index_select(0,idx).size())
        is_correct = y_test[p].float().index_select(0,idx) == torch.squeeze(torch.max(pred, 1)[1].float()).data.index_select(0,idx)
        accuracies[p][0] = int(torch.sum(is_correct.index_select(0, idx)))
        accuracies[p][1] = y_test[p].index_select(0, idx).size(0)

        # print('Accuracy', p)
        # print('Actual:', y_test[p].index_select(0, idx)[0])
        # print('Predicted:', torch.round(torch.max(pred[p], 0)[0]))
        # print('Accuracies', accuracies[p][0], accuracies[p][1])

        idx = torch.range(1, pred.size(0)).long() - 1
        var1 = pred
        if torch.cuda.is_available():
            idx = idx.cuda()
            pred = pred.cuda()

        var2 = y_test[p].index_select(0, idx).long()
        var2 = Variable(var2, requires_grad=False)

        loss = loss_fn(var1, var2)

    return loss, accuracies


def train2(rnn, lin, X_train, y_train, lens_train, X_valid, y_valid, lens_valid, num_epochs=10, loss_fn=nn.CrossEntropyLoss(size_average=True), optimizer=None):
    train_errors = torch.zeros(num_epochs)
    valid_errors = torch.zeros(num_epochs)

    X_train = X_train.float().cuda()
    y_train = y_train.long().cuda()
    lens_train = lens_train.long().cuda()
    X_valid = X_valid.float().cuda()
    y_valid = y_valid.long().cuda()
    lens_valid = lens_valid.long().cuda()

    # split into sets for training (70%), cross_validation (30%) - outputs np ndarrays
    # X_train, y_train, lens_train, X_test, y_test, lens_test = training_valid_split(X, y, lens, percent_train=percent_train)

    # Make sequence, labels for cv set
    valid_seq, valid_labels = make_cv_sequence(X_valid, y_valid, lens_valid)

    # Set initial states
    h0 = Variable(torch.zeros(num_layers, batch_size, hidden_size), requires_grad=True)
    c0 = Variable(torch.zeros(num_layers, batch_size, hidden_size), requires_grad=True)
    # print('sizes', X_test.size(), y_test.size())
    h0_valid = Variable(torch.zeros(num_layers, X_valid.size(0), hidden_size), requires_grad=True)
    c0_valid = Variable(torch.zeros(num_layers, X_valid.size(0), hidden_size), requires_grad=True)

    if torch.cuda.is_available():
        h0 = h0.cuda()
        c0 = c0.cuda()
        h0_valid = h0_valid.cuda()
        c0_valid = c0_valid.cuda()

    # run a loop for each epoch of training
    for t in range(num_epochs):
        # create training batches (padded sequences) of size batch_size
        batch, labels = make_training_batch(X_train, y_train, lens_train, batch_size)

        y_pred = None
        if is_lstm:
            y_pred, _ = rnn(batch, (h0, c0))
        else:
            y_pred, _ = rnn(batch, h0)

        var1 = lin(y_pred.data)
        var2 = labels.data

        # print('var sizes', var1.data.size(), var2.data.size(), y)
        loss = loss_fn(var1, var2)

        # calculate test errors
        valid_pred = None
        if is_lstm:
            valid_pred, _ = rnn(valid_seq, (h0_valid, c0_valid))
        else:
            valid_pred, _ = rnn(valid_seq, h0_valid)
        

        valid_var1 = lin(valid_pred.data)
        valid_var2 = valid_labels.data
        valid_loss = loss_fn(valid_var1, valid_var2)

        # record training and test errors
        train_errors[t] = loss.data[0]
        valid_errors[t] = valid_loss.data[0]

        str1 = "%.5f" % train_errors[t]
        str2 = "%.5f" % valid_loss.data[0]
        print("{}\t steps: training loss - {}\t, testing loss - {}\t".format(t, str1, str2))
        

        # Zero the gradients before running the backward pass.
        lin.zero_grad()
        rnn.zero_grad()

        # Backward pass: compute gradient of the loss with respect to all the learnable
        # parameters of the rnn. Internally, the parameters of each Module are stored
        # in Variables with requires_grad=True, so this call will compute gradients for
        # all learnable parameters in the rnn.
        loss.backward(retain_variables=True)
        
        # Update the weights using Adam-optimized gradient descent
        optimizer.step()

    return rnn, lin, train_errors, valid_errors


if __name__ == '__main__':
    loss_fn = nn.CrossEntropyLoss(size_average=True)

    rnn = None
    lin = None

    if torch.cuda.is_available():
        rnn = nn.RNN(feature_size, hidden_size, num_layers).cuda()
        lin = nn.Linear(hidden_size, num_classes).cuda() # one output per class
        if is_lstm:
            rnn = nn.LSTM(feature_size, hidden_size, num_layers).cuda()
    else:
        rnn = nn.RNN(feature_size, hidden_size, num_layers)
        lin = nn.Linear(hidden_size, num_classes) # one output per class
        if is_lstm:
            rnn = nn.LSTM(feature_size, hidden_size, num_layers)

    o_params = []
    for param in lin.parameters():
        o_params.append(param)
    for param in lin.parameters():
        o_params.append(param)

    optimizer = torch.optim.Adam(o_params)


    if not is_LOOCV:
        print('TRAINING INFO - is_lstm? {}'.format(is_lstm))
        print('-------------')
        print('actions', train_acts)
        print('devices', train_devs)
        print('step size', step_size)
        print('num_epochs', num_epochs)
        print('batch size', batch_size)
        
        print('feature size (x,y,z)', feature_size)
        print('hidden size', hidden_size)
        print('number recurrent layers', num_layers)

        print('number of classes', num_classes)

        print('writing pickle to', PICKLE_FILE)
        print('writing accuracy + pickle to', 'acc_'+PICKLE_FILE)
        print('------------------------------------------------')
        X, y, lens = getTrainingSequences(TRAIN_USERS, step_size, actions=train_acts, devices=train_devs, path=PATH_PA)
        if torch.cuda.is_available():
            X = X.cuda()
            y = y.cuda()
            lens = lens.cuda()

        rnn, lin, train_errors, test_errors = train(rnn, lin, X, y, lens, num_epochs=num_epochs, loss_fn=loss_fn, optimizer=optimizer, percent_train=0.9)

        pickle.dump((rnn, lin, train_errors, test_errors), open(PICKLE_FILE, "wb"))
        rnn, lin, train_errors, test_errors = pickle.load(open(PICKLE_FILE, "rb"))

        # Evaluate on test users
        X_test, y_test, lens_test = getTestingSequences(TEST_USERS, step_size, actions=train_acts, devices=train_devs, path=PATH_PA)
        loss, accuracies = evaluate(rnn, lin, X_test, y_test, lens_test, loss_fn=loss_fn)

        accs = torch.sum(accuracies, 0).t()
        # print('accs', accs[0][0], accs[1][0])
        print("Test loss:", "%.5f" % loss.data[0])
        print("MIMO Accuracies:", "%.5f" % (float(accs[0][0])/float(accs[1][0])))

        pickle.dump((rnn, lin, train_errors, test_errors, accuracies), open('acc'+PICKLE_FILE, "wb"))
    
    else:
        for leave_out in ['h', 'i']:
            if torch.cuda.is_available():
                rnn = nn.RNN(feature_size, hidden_size, num_layers).cuda()
                lin = nn.Linear(hidden_size, num_classes).cuda() # one output per class
                if is_lstm:
                    rnn = nn.LSTM(feature_size, hidden_size, num_layers).cuda()
            else:
                rnn = nn.RNN(feature_size, hidden_size, num_layers)
                lin = nn.Linear(hidden_size, num_classes) # one output per class
                if is_lstm:
                    rnn = nn.LSTM(feature_size, hidden_size, num_layers)

            o_params = []
            for param in lin.parameters():
                o_params.append(param)
            for param in lin.parameters():
                o_params.append(param)

            optimizer = torch.optim.Adam(o_params)


            users = TRAIN_USERS[:]
            users.remove(leave_out)

            print('\nLOOCV TRAINING INFO - is_lstm? {}'.format(is_lstm))
            print('LOOCV? - ', is_LOOCV)
            print('-------------')
            print('TRAINING USERS - ', users)
            print('LEFT OUT - ', leave_out)
            print('-------------')
            print('actions', train_acts)
            print('devices', train_devs)
            print('step size', step_size)
            print('num_epochs', num_epochs)
            print('batch size', batch_size)
            
            print('feature size (x,y,z)', feature_size)
            print('hidden size', hidden_size)
            print('number recurrent layers', num_layers)

            print('number of classes', num_classes)

            print('writing pickle to', PICKLE_FILE)
            print('writing accuracy + pickle to', 'acc_'+PICKLE_FILE)
            print('------------------------------------------------')

            X, y, lens = getTrainingSequences(users, step_size, actions=train_acts, devices=train_devs, path=PATH_PA)
            X_valid, y_valid, lens_valid = getTestingSequences([leave_out], step_size, actions=train_acts, devices=train_devs, path=PATH_PA)
            if torch.cuda.is_available():
                X = X.cuda()
                y = y.cuda()
                lens = lens.cuda()

            rnn, lin, train_errors, test_errors = train2(rnn, lin, X, y, lens, X_valid, y_valid, lens_valid, num_epochs=num_epochs, loss_fn=loss_fn, optimizer=optimizer)

            pickle.dump((rnn, lin, train_errors, test_errors), open('out_{}_'.format(leave_out)+PICKLE_FILE, "wb"))
            rnn, lin, train_errors, test_errors = pickle.load(open('out_{}_'.format(leave_out)+PICKLE_FILE, "rb"))

            # Evaluate on test users
            X_test, y_test, lens_test = getTestingSequences(TEST_USERS, step_size, actions=train_acts, devices=train_devs, path=PATH_PA)
            loss, accuracies = evaluate(rnn, lin, X_test, y_test, lens_test, loss_fn=loss_fn)

            accs = torch.sum(accuracies, 0).t()
            # print('accs', accs[0][0], accs[1][0])
            print("Test loss:", "%.5f" % loss.data[0])
            print("MIMO Accuracies:", "%.5f" % (float(accs[0][0])/float(accs[1][0])))

            pickle.dump((rnn, lin, train_errors, test_errors, accuracies), open('acc_out_{}_'.format(leave_out)+PICKLE_FILE, "wb"))
    