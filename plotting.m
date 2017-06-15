% fid = fopen('activity_recognition/anexus4_1Phones_accelerometer_bike.csv')
% out = textscan(fid,'%f %f %f %f %f %f %s %s %s','delimiter',',');
% fclose(fid);
% 
% x = out{4};
% y = out{5};
% z = out{6};
out = csvread('matlab_test.csv');

hold on
grid on

x = out(:,1);
y = out(:,2);
z = out(:,3);


plot(x)
plot(1:10:size(x),x(1:10:end))