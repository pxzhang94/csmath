
I = (imread('imgs/wall1big.jpg'));
S = tsmooth(I, 0.01, 3, 0.02, 3); 
figure, imshow(I), figure, imshow(S);

% I = (imread('imgs/wall2.jpg'));
% S = tsmooth(I, 0.01, 3, 0.02, 5); 
% figure, imshow(I), figure, imshow(S);

% I = (imread('imgs/mji.jpg'));
% S = tsmooth(I, 0.01, 3, 0.02, 5); 
% figure, imshow(I), figure, imshow(S);



