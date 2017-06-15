from action_rnn import *
import random

bootstraps = 80
FILES = ['acc_out_a_run9_model.p', 'acc_out_b_run9_model.p', 'acc_out_c_run9_model.p', 'acc_out_e_run9_model.p', 'acc_out_g_run9_model.p', 'acc_out_h_run9_model.p', 'acc_out_i_run9_model.p']

boot_accuracies = torch.zeros(bootstraps)

X_test, y_test, lens_test = getTestingSequences(TEST_USERS, step_size, actions=train_acts, devices=train_devs, path=PATH_PA)

for file in FILES:
  rnn, lin, train_errors, test_errors, accuracies = pickle.load(open(file, "rb"))

  X_test, y_test, lens_test = X_test.float(), y_test.float(), lens_test.float()

  if torch.cuda.is_available():
      X_test = X_test.cuda()
      y_test = y_test.cuda()
      lens_test = lens_test.cuda()

  for i in range(bootstraps):
    # print(X_test.size(0))
    idx = torch.zeros(X_test.size(0)) # sorting maintains the order of the lengths of sequences
    for j in range(X_test.size(0)):
      
      val = torch.randperm(X_test.size(0))[0]
      idx[int(j)] = val
    if torch.cuda.is_available():
      idx = idx.cuda()

    train_idx, _ = torch.sort(idx, 0)
    train_idx = train_idx.long()

    X = torch.index_select(X_test, 0, train_idx)
    y = torch.index_select(y_test, 0, train_idx)
    lens = torch.index_select(lens_test, 0, train_idx).float()

    # print('Sizes I', X.size(), y.size(), lens.size())

    # batch, labels = make_cv_sequence(X, y, lens)
    loss, accuracies = evaluate(rnn, lin, X, y, lens, loss_fn=nn.CrossEntropyLoss(size_average=True), show=False)

    accs = torch.sum(accuracies, 0).t()
    # print('accs', accs[0][0], accs[1][0])
    # print("Test loss:", "%.5f" % loss.data[0])
    mimo_acc = float(accs[0][0])/float(accs[1][0])
    # print("MIMO Accuracies:", "%.5f" % mimo_acc)

    boot_accuracies[i] = mimo_acc

  conf_int, _ = torch.sort(boot_accuracies, 0)
  # compute 95% conf interval
  conf_int = conf_int[2:-2]
  
  str1 = "%.5f" % torch.min(conf_int,0)[0][0]
  str2 = "%.5f" % torch.max(conf_int,0)[0][0]
  print('Confidence Intervals for {}: [{}, {}]'.format(file, str1, str2))

