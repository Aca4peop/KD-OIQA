% example of convert OIQA database to viewport images
clear, clc;
% the dir path you want to save the viewport images
writeDataPath = 'D:\temp\OIQA\cubic';
if exist(writeDataPath, 'dir') == 0
    mkdir(writeDataPath);
end
directions = {'F', 'R', 'BA', 'L', 'T', 'BO'};
% the root directory path of the original ERP images of OIQA database
dataRootPath = 'D:\dataset\OIQA\distorted_images\';

for i = 1:320
    img_path = [dataRootPath, 'img', num2str(i), '.jpg'];
    if exist(img_path, 'file') == 0
        img_path = [dataRootPath, 'img', num2str(i), '.png'];
    end
    img = imread(img_path);
    out = equi2cubic(img);
    for idx = 1:numel(directions)
        imwrite(out{idx}, [writeDataPath, '\', num2str(i, "%03d"), directions{idx}, '.jpg']);
    end
end