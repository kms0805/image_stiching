%% ICV Assignment#3

run('vlfeat-0.9.21/toolbox/vl_setup')
%% 이미지 전처리
clear;
clc;
I_1 = imread('j1.jpg');
I_2 = imread('j2.jpg');
I_3 = imread('j3.jpg');
I_1 = imrotate(I_1,-90);
I_2 = imrotate(I_2,-90);
I_3 = imrotate(I_3,-90);
I_1 = imresize(I_1,[256 256]);
I_2 = imresize(I_2,[256 256]);
I_3 = imresize(I_3,[256 256]);
figure(1)
subplot(1,3,1);
imshow(I_1);
subplot(1,3,2);
imshow(I_2);
subplot(1,3,3);
imshow(I_3);

%% Feature Extraction
I1 = single(rgb2gray(I_1));
I2 = single(rgb2gray(I_2));
I3 = single(rgb2gray(I_3));
[F1, D1] = vl_sift(I1,'PeakThresh', 2.25);
[F2, D2] = vl_sift(I2,'PeakThresh', 2.25);
[F3, D3] = vl_sift(I3,'PeakThresh', 2.25);

%% Feature Matching
[matches12, scores12] = vl_ubcmatch(D1,D2);
[matches32, scores32] = vl_ubcmatch(D3,D2);
figure(1)
displaymatches(I1,I2,F1,F2,matches12)
figure(2)
displaymatches(I3,I2,F3,F2,matches32)

%% Homography Estimation using RANSAC
H12 = HbyRANSAC(F1,F2,matches12);
H32 = HbyRANSAC(F3,F2,matches32);
%% Warping pair Images

[img12, idx12] = pairwarp(H12,I_1,I_2);
figure(1)
imshow(img12);
[img32,idx32] = pairwarp(H32,I_3,I_2);
figure(2)
imshow(img32);

%% Warping three Images

figure(3)
img123 = threewarp(H12, H32, I_1,I_2, I_3);
imshow(img123)

%% total distance
n = size(matches12,2);
dist12 = zeros(n,1);
x1 = [F1(1,matches12(1,:)); F1(2,matches12(1,:)); ones(1,n)];
x2 = [F2(1,matches12(2,:)); F2(2,matches12(2,:)); ones(1,n)];
 for i = 1:n
        d_tmp1 = inv(H12)*x2(:,i);
        d_tmp1 = d_tmp1/d_tmp1(3);
        dif1 = (x1(:,i) - d_tmp1).^2;
        d_tmp2 = H12*x1(:,i);
        d_tmp2 = d_tmp2/d_tmp2(3);
        dif2 = (x2(:,i) - d_tmp2).^2;
        dist12(i) = sum(sqrt(dif1))+sum(sqrt(dif2)); %compute distance 
 end 
 
 n = size(matches32,2);
dist32 = zeros(n,1);
x1 = [F3(1,matches32(1,:)); F3(2,matches32(1,:)); ones(1,n)];
x2 = [F2(1,matches32(2,:)); F2(2,matches32(2,:)); ones(1,n)];
 for i = 1:n
        d_tmp1 = inv(H32)*x2(:,i);
        d_tmp1 = d_tmp1/d_tmp1(3);
        dif1 = (x1(:,i) - d_tmp1).^2;
        d_tmp2 = H32*x1(:,i);
        d_tmp2 = d_tmp2/d_tmp2(3);
        dif2 = (x2(:,i) - d_tmp2).^2;
        dist32(i) = sum(sqrt(dif1))+sum(sqrt(dif2)); %compute distance 
 end
figure(1)
plot(dist12)
title("dist12")
figure(2)
plot(dist32)
title("dist32")
%% functions

%Display the detected correspondences using lines
function d = displaymatches(I1,I2,F1,F2,matches)
%display 2 image first
catimg = horzcat(uint8(I1), uint8(I2));
imshow(catimg);
hold on;
%detected correspondences
x1 = F1(1,matches(1,:));
y1 = F1(2,matches(1,:));
x2 = F2(1,matches(2,:)) + size(I1,2);
y2 = F2(2,matches(2,:));
%detected correspondences를 잇는 선들을 표시
h = line([x1; x2], [y1; y2]);
set(h,'linewidth', 0.5, 'color', 'b');
%detected correspondences를 표시
vl_plotframe(F1(:,matches(1,:)));
F2_temp= F2;
F2_temp(1,:) = F2(1,:) + size(I1,2);
vl_plotframe(F2_temp(:,matches(2,:)));
end

%HbyRANSAC
function H = HbyRANSAC(F1,F2,matches)
%initialize
max_inliers = 0;
n = size(matches,2);
N=inf;
sample_count=0;
s = 4;
p = 0.99;
t = 1.25;
goodH = zeros(3);      %best H of iteration
gooddist = zeros(n,1); %best H's dist

while N>sample_count
    num_inliers = 0;
    
    %DLT
    r = randsample(n,4);
    x1 = [F1(1,matches(1,r)); F1(2,matches(1,r)); 1 1 1 1];
    x2 = [F2(1,matches(2,r)); F2(2,matches(2,r)); 1 1 1 1]; %4쌍의correspondences
    x1t = x1'; 
    A = [0 0 0 x1t(1,:) -1*x2(2,1)*x1t(1,:);
        x1t(1,:) 0 0 0 -1*x2(1,1)*x1t(1,:);
        0 0 0 x1t(2,:) -1*x2(2,2)*x1t(2,:);
        x1t(2,:) 0 0 0 -1*x2(1,2)*x1t(2,:);
        0 0 0 x1t(3,:) -1*x2(2,3)*x1t(3,:);
        x1t(3,:) 0 0 0 -1*x2(1,3)*x1t(3,:);
        0 0 0 x1t(4,:) -1*x2(2,4)*x1t(4,:);
        x1t(4,:) 0 0 0 -1*x2(1,4)*x1t(4,:)];
    [U,S,V]=svd(A);
    h = V(:,end);
    H = reshape(h,[],3);
    H = H';
    
    %RANSAC
    x1 = [F1(1,matches(1,:)); F1(2,matches(1,:)); ones(1,n)];
    x2 = [F2(1,matches(2,:)); F2(2,matches(2,:)); ones(1,n)];
    dist = zeros(n,1);
    for i = 1:n
        d_tmp1 = inv(H)*x2(:,i);
        d_tmp1 = d_tmp1/d_tmp1(3);
        dif1 = (x1(:,i) - d_tmp1).^2;
        d_tmp2 = H*x1(:,i);
        d_tmp2 = d_tmp2/d_tmp2(3);
        dif2 = (x2(:,i) - d_tmp2).^2;
        dist(i) = sum(sqrt(dif1))+sum(sqrt(dif2)); %compute distance 
        if (dist(i) < t)
             num_inliers = num_inliers + 1; %Compute the number of inliers
        end
    end
    if(max_inliers < num_inliers) %찾은 H가 이때까지 중 best result라면 갱신
        max_inliers = num_inliers;
        goodH = H;
        gooddist =dist;
    end
    % Adaptive determination of the # of samples for RANSAC
     e = 1 - num_inliers/n;
     N = log(1-p)/log(1-(1-e)^s);
     sample_count = sample_count + 1;
end
% re-estimation with all inliers using DLT while convergence
now = 0;
while(num_inliers ~= now)
    num_inliers = now;
    now = 0; 
    k = find(dist<1.25);
    in_x1 = [F1(1,matches(1,k)); F1(2,matches(1,k)); ones(1,size(k,1))];
    in_x2 = [F2(1,matches(2,k)); F2(2,matches(2,k)); ones(1,size(k,1))];
    in_x1t = in_x1';
    A=[];
    for i = 1:size(k,1)
        A=[A;0 0 0 in_x1t(i,:) -1*in_x2(2,i)*in_x1t(i,:);
            in_x1t(i,:) 0 0 0 -1*in_x2(1,i)*in_x1t(i,:)];
    end
    [U,S,V]=svd(A);
    h = V(:,end);
    H = reshape(h,[],3);
    H = H';
    dist = zeros(n,1);
    for i = 1:n
        d_tmp1 = inv(H)*x2(:,i);
        d_tmp1 = d_tmp1/d_tmp1(3);
        dif1 = (x1(:,i) - d_tmp1).^2;
        d_tmp2 = H*x1(:,i);
        d_tmp2 = d_tmp2/d_tmp2(3);
        dif2 = (x2(:,i) - d_tmp2).^2;
        dist(i) = sum(sqrt(dif1))+sum(sqrt(dif2)); %compute distance 
        if (dist(i) < t)
             now = now + 1; %Compute the number of inliers
        end
    end
end
end

%pair warping
function [img,idx] = pairwarp(H,I_1,I_2)
%I_2의 원래 (1,1)에 있던 점이 어느 index에 있는지를 파악
idx = [1; 1];
T = maketform('projective',H');
res = imtransform(I_1,T);
%%warping other images to the plane of I2

%compute corner location after warping
corner1 = H * [0;0;1];
corner2 = H * [size(I_1,2);0;1];
corner3 = H * [0;size(I_1,1);1];
corner4 = H * [size(I_1,2);size(I_1,1);1];
corner1 = corner1/corner1(3);
corner2 = corner2/corner2(3);
corner3 = corner3/corner3(3);
corner4 = corner4/corner4(3);
corner_x = [corner1(1);corner2(1);corner3(1);corner4(1)];
corner_y = [corner1(2);corner2(2);corner3(2);corner4(2)];

%transformed 되었을 때의 corner들의 위치를 이용하여 이미지 위치 이동 
minc_x = min(corner_x); % warping된 코너 중 가장 왼쪽
minc_y = min(corner_y); % warping된 코너 중 가장 위쪽
maxc_x = max(corner_x); % warping된 코너 중 가장 오른쪽
maxc_y = max(corner_y); % warping된 코너 중 가장 아래쪽
newI1 = imtranslate(res,[minc_x,minc_y], 'OutputView', 'full');
newI2 = imtranslate(I_2,[-min(minc_x,0),-min(minc_y,0)], 'OutputView', 'full');
idx(1) = idx(1) - min(minc_y,0);
idx(2) = idx(2) - min(minc_x,0);
if maxc_x > size(I_1,2)
    newI2 = imtranslate(newI2,[-(maxc_x-size(I_1,2)),0], 'OutputView', 'full');
end
if maxc_y > size(I_1,1)
    newI2 = imtranslate(newI2,[0,-(maxc_y-size(I_1,1))], 'OutputView', 'full');
end
for i = 1:size(newI2,1)
    for j = 1:size(newI2,2)
        if sum(newI2(i,j,:)) < sum(newI1(i,j,:)) 
             newI2(i,j,:) = newI1(i,j,:);
        end         
    end
end
img = newI2;
end

%three image warp
function img123 = threewarp(H12, H32, I_1,I_2, I_3)
[img12,idx12]=pairwarp(H12,I_1,I_2);
[img32,idx32]=pairwarp(H32,I_3,I_2);
if idx12(1)>idx32(1)
    img32 = imtranslate(img32,[0,idx12(1)-idx32(1)], 'OutputView', 'full');
else
    img12 = imtranslate(img12,[0,idx32(1)-idx12(1),0], 'OutputView', 'full');
end
if idx12(2)>idx32(2)
    img32 = imtranslate(img32,[idx12(2)-idx32(2),0], 'OutputView', 'full');
else
    img12 = imtranslate(img12,[idx32(2)-idx12(2),0], 'OutputView', 'full');
end
r = max(size(img12,1),size(img32,1));
c = max(size(img12,2),size(img32,2));

img12 = imtranslate(img12,[size(img12,2)-c,size(img12,1)-r], 'OutputView', 'full');
img32 = imtranslate(img32,[size(img32,2)-c,size(img32,1)-r], 'OutputView', 'full');

for i = 1:size(img32,1)
    for j = 1:size(img32,2)
        if sum(img32(i,j,:)) < sum(img12(i,j,:)) 
            img32(i,j,:) = img12(i,j,:);
        end         
    end
end
img123 = img32;
end
