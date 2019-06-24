function Prepare_TrainData_HR_LR_DN()
%% settings
path_save = '../dataset/DIV2K/GreyNoise/N';
path_src = '../dataset/DIV2K/DIV2K_train_HR/';
path_grey = '../dataset/DIV2K/Grey/';

if ~exist(path_grey)
    mkdir(path_grey)
end

filepaths = dir(fullfile(path_src, '*.png'));

nb_im = length(filepaths);

for i = 1:nb_im 
    name = filepaths(i).name;
    ImHR = imread([path_src name]);
    im = rgb2gray(ImHR);
    imwrite(uint8(im), [path_grey name]);
end

snr = {10, 30, 50, 70}; %Noise level
for i = 1:length(snr)
    save_folder = [path_save num2str(snr{i})];

    if ~exist(save_folder)
        mkdir(save_folder)
    end

    for IdxIm = 1:nb_im
        name = filepaths(IdxIm).name;
        ImHR = imread([path_grey filepaths(IdxIm).name]);
        im_nr = single(ImHR) + single(snr{i}*randn(size(ImHR)));
        
        % save image
        imwrite(uint8(im_nr), [save_folder '/' name(1:end-4) '_N' num2str(snr{i}) '.png']);
    end 
end

