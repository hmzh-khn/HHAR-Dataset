%%
set(gca,'FontSize',14,'fontWeight','bold')
set(findall(gcf,'type','text'),'FontSize',14,'fontWeight','bold')

% run4.txt was without an LSTM, using 3 recurrent layers, not getting
% better, so I ended it early.
%

filename = 'run4.txt';
fileID = fopen(filename);
C4 = textscan(fileID,'%f	 steps: training loss - %f	, testing loss - %f');
fclose(fileID);
% celldisp(C)
hold on
grid on
plot(C4{1}, C4{2})
plot(C4{1}, C4{3})
hold off
title('Loss from training an RNN with 3 recurrent layers for 400 epochs to classify 6 actions(no downsampling)')
xlabel('number of epochs')
ylabel('loss')
legend('training loss', 'test loss')

%%
% Used this for validation of LSTM v. RNN, failed early for some reason
figure(2)
set(gca,'FontSize',14,'fontWeight','bold')
set(findall(gcf,'type','text'),'FontSize',14,'fontWeight','bold')
filename = 'run5data.txt';
fileID = fopen(filename);
C5 = textscan(fileID,'%f	 steps: training loss - %f	, testing loss - %f');
fclose(fileID);
celldisp(C5)
hold on
grid on
plot(C5{1}, C5{2})
plot(C5{1}, C5{3})
title('Loss from training an LSTM with 2 recurrent layers for 527 epochs to classify 6 actions (no downsampling)')
xlabel('number of epochs')
ylabel('loss')
legend('training loss', 'test loss')

%%
% run 6
figure(3)
set(gca,'FontSize',14,'fontWeight','bold')
set(findall(gcf,'type','text'),'FontSize',14,'fontWeight','bold')
filename = 'run6data.txt';
fileID = fopen(filename);
C5 = textscan(fileID,'%f	 steps: training loss - %f	, testing loss - %f');
fclose(fileID);
celldisp(C5)
hold on
grid on
plot(C5{1}, C5{2})
plot(C5{1}, C5{3})
title('LSTM Loss for binary classification between bike and sit actions (2 recurrent layers w/ 10x downsampling - 5Hz)')
xlabel('number of epochs')
ylabel('loss')
legend('training loss', 'test loss')

%%
% run 8
figure(4)
set(gca,'FontSize',14,'fontWeight','bold')
set(findall(gcf,'type','text'),'FontSize',14,'fontWeight','bold')
filename = 'run8data.txt';
fileID = fopen(filename);
C5 = textscan(fileID,'%f	 steps: training loss - %f	, testing loss - %f');
fclose(fileID);
celldisp(C5)
hold on
grid on
plot(C5{1}, C5{2})
plot(C5{1}, C5{3})
title('RNN Loss for binary classification between bike and sit actions (2 recurrent layers w/ 10x downsampling - 5Hz)')
xlabel('number of epochs')
ylabel('loss')
legend('training loss', 'test loss')

% run 9a-i?
