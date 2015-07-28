function  Experiment(data_file)
% Experiment: the main function used to compare all the online
% algorithms
%--------------------------------------------------------------------------
% Input:
%      dataset_name, name of the dataset, e.g. 'birds-food'
%
% Output:
%      a table containing the accuracies, the numbers of support vectors,
%      the running times of all the online learning algorithms on the
%      inputed datasets
%      a figure for the online average accuracies of all the online
%      learning algorithms
%      a figure for the online numbers of SVs of all the online learning
%      algorithms
%      a figure for the online running time of all the online learning
%      algorithms
%--------------------------------------------------------------------------


%load dataset
similarity_method = 'pearson';
load(sprintf('../data/original/%d', data_file));
load('../data/ID');
load(sprintf('../data/similarity/%s/%d', similarity_method, data_file));

m = size(ID, 2);

% image data
image_Y = image_gnd(1:m,:);
image_Y = full(image_Y);
image_X = image_fea(1:m,:);
image_X = [image_X; co_image_fea];
[image_n, image_d] = size(image_X);

% text data
text_Y = text_gnd(1:m,:);
text_Y = full(text_Y);
text_X = text_fea(1:m,:);
text_X = [text_X; co_text_fea];
[text_n, text_d] = size(text_X);

% set parameters
options.C = 5;
options.sigma = 1;
options.sigma2 = 8;
options.t_tick = round(size(ID,2)/10);
options.K = 10;
options.m = m;
options.eta = 0.01;

options.gamma_1_1 = 0.5;
options.gamma_2_1 = 0.5;
options.alpha_1_1 = 0.5;
options.alpha_2_1 = 0.5;

options.gamma_1_2 = 0.5;
options.gamma_2_2 = 0.5;
options.alpha_1_2 = 0.5;
options.alpha_2_2 = 0.5;

options.gamma1 = 1;
options.gamma2 = 1;

%%
% scale
MaxX=max(image_X,[],2);
MinX=min(image_X,[],2);
DifX=MaxX-MinX;
idx_DifNonZero=(DifX~=0);
DifX_2=ones(size(DifX));
DifX_2(idx_DifNonZero,:)=DifX(idx_DifNonZero,:);
image_X = bsxfun(@minus, image_X, MinX);
image_X = bsxfun(@rdivide, image_X , DifX_2);

MaxX=max(text_X,[],2);
MinX=min(text_X,[],2);
DifX=MaxX-MinX;
idx_DifNonZero=(DifX~=0);
DifX_2=ones(size(DifX));
DifX_2(idx_DifNonZero,:)=DifX(idx_DifNonZero,:);
text_X = bsxfun(@minus, text_X, MinX);
text_X = bsxfun(@rdivide, text_X , DifX_2);


% kernel
P = sum(image_X.*image_X,2);
P = full(P);
image_kernel = exp(-(repmat(P',image_n,1) + repmat(P,1,image_n)- 2*image_X*image_X')/(2*options.sigma^2));
P = sum(text_X.*text_X,2);
P = full(P);
text_kernel = exp(-(repmat(P',text_n,1) + repmat(P,1,text_n)- 2*text_X*text_X')/(2*options.sigma^2));
%image_kernel = image_X * image_X';
%text_kernel = text_X * text_X';

%% run experiments:
for i=1:size(ID,1),
   % fprintf(1,'running on the %d-th trial...\n',i);
    id_list = ID(i, :);

		%1. personal model
    [classifier, err_count, run_time, mistakes] = PA1(image_Y,image_kernel,options,id_list);
    %[classifier, err_count, run_time, mistakes] = PA1_linear(image_X, image_Y,options,id_list);
    err_PA1_personal_image(i) = err_count;
    time_PA1_personal_image(i) = run_time;
    mistakes_list_PA1_personal_image(i,:) = mistakes;

    [classifier, err_count, run_time, mistakes] = PA1(text_Y,text_kernel,options,id_list);
    %[classifier, err_count, run_time, mistakes] = PA1_linear(text_X, text_Y,options,id_list);
    err_PA1_personal_text(i) = err_count;
    time_PA1_personal_text(i) = run_time;
    mistakes_list_PA1_personal_text(i,:) = mistakes;
		
		%2. shared-loss model
    [classifier_image, err_count_image, mistakes_image, classifier_text, err_count_text, mistakes_text, run_time] = PA1_shared(image_Y,image_kernel,text_Y,text_kernel,options,id_list);
    %[classifier_image, err_count_image, mistakes_image, classifier_text, err_count_text, mistakes_text, run_time] = PA1_shared_linear(image_X, image_Y, text_X, text_Y,options,id_list);
    err_PA1_shared_image(i) = err_count_image;
    mistakes_list_PA1_shared_image(i,:) = mistakes_image;
    err_PA1_shared_text(i) = err_count_text;
    mistakes_list_PA1_shared_text(i,:) = mistakes_text;
    time_PA1_shared(i) = run_time;

		%3. domain-specific (1) + optimization
		[classifier_image, err_count_image, mistakes_image, classifier_text, err_count_text, mistakes_text, run_time] = dso1(image_kernel, image_Y, cii', text_kernel, text_Y, ctt', options, id_list);
		%[classifier_image, err_count_image, mistakes_image, classifier_text, err_count_text, mistakes_text, run_time] = dso1_linear(image_X(1:m,:), image_Y, cii', co_image_fea, text_X(1:m,:), text_Y, ctt', co_text_fea, options, id_list);
    err_dso1_image(i) = err_count_image;
    mistakes_list_dso1_image(i,:) = mistakes_image;
    err_dso1_text(i) = err_count_text;
    mistakes_list_dso1_text(i,:) = mistakes_text;
    time_dso1(i) = run_time;
		
		%4. PA1_extra
		[classifier_image, err_count_image, mistakes_image, classifier_text, err_count_text, mistakes_text, run_time] = PA1_extra(image_kernel, image_Y, cii', text_kernel, text_Y, ctt', options, id_list);
    err_PA1_extra_image(i) = err_count_image;
    mistakes_list_PA1_extra_image(i,:) = mistakes_image;
    err_PA1_extra_text(i) = err_count_text;
    mistakes_list_PA1_extra_text(i,:) = mistakes_text;
    time_PA1_extra(i) = run_time;

		%5. PA1_extra_sim
		[classifier_image, err_count_image, mistakes_image, classifier_text, err_count_text, mistakes_text, run_time] = PA1_extra_sim(image_kernel, image_Y, cii', text_kernel, text_Y, ctt', options, id_list);
    err_PA1_extra_sim_image(i) = err_count_image;
    mistakes_list_PA1_extra_sim_image(i,:) = mistakes_image;
    err_PA1_extra_sim_text(i) = err_count_text;
    mistakes_list_PA1_extra_sim_text(i,:) = mistakes_text;
    time_PA1_extra_sim(i) = run_time;
		
		%4. PA1_extra_self
		[classifier_image, err_count_image, mistakes_image, classifier_text, err_count_text, mistakes_text, run_time] = PA1_extra_self(image_kernel, image_Y, cii', text_kernel, text_Y, ctt', options, id_list);
    err_PA1_extra_self_image(i) = err_count_image;
    mistakes_list_PA1_extra_self_image(i,:) = mistakes_image;
    err_PA1_extra_self_text(i) = err_count_text;
    mistakes_list_PA1_extra_self_text(i,:) = mistakes_text;
    time_PA1_extra_self(i) = run_time;

		%5. PA1_extra_sim_self
		[classifier_image, err_count_image, mistakes_image, classifier_text, err_count_text, mistakes_text, run_time] = PA1_extra_sim_self(image_kernel, image_Y, cii', text_kernel, text_Y, ctt', options, id_list);
    err_PA1_extra_sim_self_image(i) = err_count_image;
    mistakes_list_PA1_extra_sim_self_image(i,:) = mistakes_image;
    err_PA1_extra_sim_self_text(i) = err_count_text;
    mistakes_list_PA1_extra_sim_self_text(i,:) = mistakes_text;
    time_PA1_extra_sim_self(i) = run_time;
		
		%6. latent + optimization
		%[classifier_image, err_count_image, mistakes_image, classifier_text, err_count_text, mistakes_text, run_time] = lo(image_Y, image_kernel, text_Y, text_kernel, options, id_list);
    %err_lo_image(i) = err_count_image;
    %mistakes_list_lo_image(i,:) = mistakes_image;
    %err_lo_text(i) = err_count_text;
    %mistakes_list_lo_text(i,:) = mistakes_text;
    %time_lo(i) = run_time;
		
		%7. Online Gradient Descent algorithm
    [classifier, err_count, run_time, mistakes] = OGD(image_Y, image_kernel, options, id_list);
		err_OGD_personal_image(i) = err_count;
		time_OGD_personal_image(i) = run_time;
		mistakes_list_OGD_personal_image(i,:) = mistakes;

    [classifier, err_count, run_time, mistakes] = OGD(text_Y, text_kernel, options, id_list);
		err_OGD_personal_text(i) = err_count;
		time_OGD_personal_text(i) = run_time;
		mistakes_list_OGD_personal_text(i,:) = mistakes;

    %8. Heterogeneous Online Transfer Learning
		[classifier, err_count, run_time, mistakes] = HetOTL(image_Y, image_kernel, text_kernel, options, id_list, cii');
		err_HetOTL_image(i) = err_count;
		time_HetOTL_image(i) = run_time;
		mistakes_list_HetOTL_image(i,:) = mistakes;

		[classifier, err_count, run_time, mistakes] = HetOTL(text_Y, text_kernel, image_kernel, options, id_list, ctt');
		err_HetOTL_text(i) = err_count;
		time_HetOTL_text(i) = run_time;
		mistakes_list_HetOTL_text(i,:) = mistakes;

    %9. Heterogeneous Online Transfer Learning Shared
		[classifier_1, err_count_1, mistakes_1, classifier_2, err_count_2, mistakes_2, run_time] = HetOTL_shared(image_kernel, image_Y, cii', text_kernel, text_Y, ctt', options, id_list);
		err_HetOTL_shared_image(i) = err_count_1;
		time_HetOTL_shared_image(i) = run_time;
		mistakes_list_HetOTL_shared_image(i,:) = mistakes_1;
		err_HetOTL_shared_text(i) = err_count_2;
		time_HetOTL_shared_text(i) = run_time;
		mistakes_list_HetOTL_shared_text(i,:) = mistakes_2;

    %10. PA1 feature
		[classifier, err_count, run_time, mistakes] = PA1_fea(image_Y, text_kernel, cii', options, id_list);
		err_PA1_fea_image(i) = err_count;
		time_PA1_fea_image(i) = run_time;
		mistakes_list_PA1_fea_image(i,:) = mistakes;

		[classifier, err_count, run_time, mistakes] = PA1_fea(text_Y, image_kernel, ctt', options, id_list);
		err_PA1_fea_text(i) = err_count;
		time_PA1_fea_text(i) = run_time;
		mistakes_list_PA1_fea_text(i,:) = mistakes;


		%7. domain-specific (2) + latent + optimization
		%[classifier_image, err_count_image, mistakes_image, classifier_text, err_count_text, mistakes_text, run_time] = dslo2(image_Y, image_kernel, text_Y, text_kernel, options, id_list);
    %err_dslo2_image(i) = err_count_image;
    %mistakes_list_dslo2_image(i,:) = mistakes_image;
    %err_dslo2_text(i) = err_count_text;
    %mistakes_list_dslo2_text(i,:) = mistakes_text;
    %time_dslo2(i) = run_time;
		
    
    %1. PA-I
    %[classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = PA1(Y2,K2,options,ID);
    %nSV_PA1(i) = length(classifier.SV);
    %err_PA1(i) = err_count;
    %time_PA1(i) = run_time;
    %mistakes_list_PA1(i,:) = mistakes;
    %SVs_PA1(i,:) = SVs;
    %TMs_PA1(i,:) = TMs;
    
    
end


stat_file = sprintf('../stat/%s/%d-stat', similarity_method, data_file);
save(stat_file, 'err_PA1_personal_image', 'time_PA1_personal_image', 'mistakes_list_PA1_personal_image', 'err_PA1_personal_text', 'time_PA1_personal_text', 'mistakes_list_PA1_personal_text', 'err_OGD_personal_image', 'time_OGD_personal_image', 'mistakes_list_OGD_personal_image', 'err_OGD_personal_text', 'time_OGD_personal_text', 'mistakes_list_OGD_personal_text', 'err_PA1_shared_image', 'mistakes_list_PA1_shared_image', 'err_PA1_shared_text', 'mistakes_list_PA1_shared_text', 'time_PA1_shared', 'err_dso1_image', 'mistakes_list_dso1_image', 'err_dso1_text', 'time_dso1', 'mistakes_list_dso1_text', 'err_PA1_extra_image', 'mistakes_list_PA1_extra_image', 'err_PA1_extra_text', 'time_PA1_extra', 'mistakes_list_PA1_extra_text', 'err_PA1_extra_sim_image', 'mistakes_list_PA1_extra_sim_image', 'err_PA1_extra_sim_text', 'mistakes_list_PA1_extra_sim_text', 'time_PA1_extra_sim', 'err_HetOTL_image', 'time_HetOTL_image', 'mistakes_list_HetOTL_image', 'err_HetOTL_text', 'time_HetOTL_text', 'mistakes_list_HetOTL_text', 'err_HetOTL_shared_image', 'time_HetOTL_shared_image', 'mistakes_list_HetOTL_shared_image', 'err_HetOTL_shared_text', 'time_HetOTL_shared_text', 'mistakes_list_HetOTL_shared_text', 'err_PA1_fea_image', 'time_PA1_fea_image', 'mistakes_list_PA1_fea_image', 'err_PA1_fea_text', 'time_PA1_fea_text', 'mistakes_list_PA1_fea_text');

fprintf(1,'-------------------------------------------------------------------------------\n');
fprintf(1,'                  number of mistakes,        cpu running time\n');
fprintf(1,'PA1_personal_image \t %.4f %.4f \t %.4f %.4f\n', mean(err_PA1_personal_image)/m*100,  std(err_PA1_personal_image)/m*100, mean(time_PA1_personal_image)/m*100, std(time_PA1_personal_image));
fprintf(1,'PA1_personal_text \t %.4f %.4f \t %.4f %.4f\n', mean(err_PA1_personal_text)/m*100,  std(err_PA1_personal_text)/m*100, mean(time_PA1_personal_text)/m*100, std(time_PA1_personal_text));
fprintf(1,'PA1_shared_image \t %.4f %.4f \t %.4f %.4f\n', mean(err_PA1_shared_image)/m*100,  std(err_PA1_shared_image)/m*100, mean(time_PA1_shared)/m*100, std(time_PA1_shared));
fprintf(1,'PA1_shared_text \t %.4f %.4f \t %.4f %.4f\n', mean(err_PA1_shared_text)/m*100,  std(err_PA1_shared_text)/m*100, mean(time_PA1_shared)/m*100, std(time_PA1_shared));
fprintf(1,'dso1_image \t %.4f %.4f \t %.4f %.4f\n', mean(err_dso1_image)/m*100,  std(err_dso1_image)/m*100, mean(time_dso1)/m*100, std(time_dso1));
fprintf(1,'dso1_text \t %.4f %.4f \t %.4f %.4f\n', mean(err_dso1_text)/m*100,  std(err_dso1_text)/m*100, mean(time_dso1)/m*100, std(time_dso1));
fprintf(1,'PA1_extra_image \t %.4f %.4f \t %.4f %.4f\n', mean(err_PA1_extra_image)/m*100,  std(err_PA1_extra_image)/m*100, mean(time_PA1_extra)/m*100, std(time_PA1_extra));
fprintf(1,'PA1_extra_text \t %.4f %.4f \t %.4f %.4f\n', mean(err_PA1_extra_text)/m*100,  std(err_PA1_extra_text)/m*100, mean(time_PA1_extra)/m*100, std(time_PA1_extra));
fprintf(1,'PA1_extra_sim_image \t %.4f %.4f \t %.4f %.4f\n', mean(err_PA1_extra_sim_image)/m*100,  std(err_PA1_extra_sim_image)/m*100, mean(time_PA1_extra_sim)/m*100, std(time_PA1_extra_sim));
fprintf(1,'PA1_extra_sim_text \t %.4f %.4f \t %.4f %.4f\n', mean(err_PA1_extra_sim_text)/m*100,  std(err_PA1_extra_sim_text)/m*100, mean(time_PA1_extra_sim)/m*100, std(time_PA1_extra_sim));
fprintf(1,'PA1_extra_self_image \t %.4f %.4f \t %.4f %.4f\n', mean(err_PA1_extra_self_image)/m*100,  std(err_PA1_extra_self_image)/m*100, mean(time_PA1_extra_self)/m*100, std(time_PA1_extra_self));
fprintf(1,'PA1_extra_self_text \t %.4f %.4f \t %.4f %.4f\n', mean(err_PA1_extra_self_text)/m*100,  std(err_PA1_extra_self_text)/m*100, mean(time_PA1_extra_self)/m*100, std(time_PA1_extra_self));
fprintf(1,'PA1_extra_sim_self_image \t %.4f %.4f \t %.4f %.4f\n', mean(err_PA1_extra_sim_self_image)/m*100,  std(err_PA1_extra_sim_self_image)/m*100, mean(time_PA1_extra_sim_self)/m*100, std(time_PA1_extra_sim_self));
fprintf(1,'PA1_extra_sim_self_text \t %.4f %.4f \t %.4f %.4f\n', mean(err_PA1_extra_sim_self_text)/m*100,  std(err_PA1_extra_sim_self_text)/m*100, mean(time_PA1_extra_sim_self)/m*100, std(time_PA1_extra_sim_self));
fprintf(1,'OGD_personal_image \t %.4f %.4f \t %.4f %.4f\n', mean(err_OGD_personal_image)/m*100,  std(err_OGD_personal_image)/m*100, mean(time_OGD_personal_image)/m*100, std(time_OGD_personal_image));
fprintf(1,'OGD_personal_text \t %.4f %.4f \t %.4f %.4f\n', mean(err_OGD_personal_text)/m*100,  std(err_OGD_personal_text)/m*100, mean(time_OGD_personal_text)/m*100, std(time_OGD_personal_text));
fprintf(1,'HetOTL_image \t %.4f %.4f \t %.4f %.4f\n', mean(err_HetOTL_image)/m*100,  std(err_HetOTL_image)/m*100, mean(time_HetOTL_image)/m*100, std(time_HetOTL_image));
fprintf(1,'HetOTL_text \t %.4f %.4f \t %.4f %.4f\n', mean(err_HetOTL_text)/m*100,  std(err_HetOTL_text)/m*100, mean(time_HetOTL_text)/m*100, std(time_HetOTL_text));
fprintf(1,'HetOTL_shared_image \t %.4f %.4f \t %.4f %.4f\n', mean(err_HetOTL_shared_image)/m*100,  std(err_HetOTL_shared_image)/m*100, mean(time_HetOTL_shared_image)/m*100, std(time_HetOTL_shared_image));
fprintf(1,'HetOTL_shared_text \t %.4f %.4f \t %.4f %.4f\n', mean(err_HetOTL_shared_text)/m*100,  std(err_HetOTL_shared_text)/m*100, mean(time_HetOTL_shared_text)/m*100, std(time_HetOTL_shared_text));
fprintf(1,'PA1_fea_image \t %.4f %.4f \t %.4f %.4f\n', mean(err_PA1_fea_image)/m*100,  std(err_PA1_fea_image)/m*100, mean(time_PA1_fea_image)/m*100, std(time_PA1_fea_image));
fprintf(1,'PA1_fea_text \t %.4f %.4f \t %.4f %.4f\n', mean(err_PA1_fea_text)/m*100,  std(err_PA1_fea_text)/m*100, mean(time_PA1_fea_text)/m*100, std(time_PA1_fea_text));
%fprintf(1,'dso2_image \t %.4f %.4f \t %.4f %.4f\n', mean(err_dso2_image)/m*100,  std(err_dso2_image)/m*100, mean(time_dso2_image)/m*100, std(time_dso2_image));
%fprintf(1,'dso2_text \t %.4f %.4f \t %.4f %.4f\n', mean(err_dso2_text)/m*100,  std(err_dso2_text)/m*100, mean(time_dso2_text)/m*100, std(time_dso2_text));
%fprintf(1,'lo_image \t %.4f %.4f \t %.4f %.4f\n', mean(err_lo_image)/m*100,  std(err_lo_image)/m*100, mean(time_lo_image)/m*100, std(time_lo_image));
%fprintf(1,'lo_text \t %.4f %.4f \t %.4f %.4f\n', mean(err_lo_text)/m*100,  std(err_lo_text)/m*100, mean(time_lo_text)/m*100, std(time_lo_text));
%fprintf(1,'dslo1_image \t %.4f %.4f \t %.4f %.4f\n', mean(err_dslo1_image)/m*100,  std(err_dslo1_image)/m*100, mean(time_dslo1_image)/m*100, std(time_dslo1_image));
%fprintf(1,'dslo1_text \t %.4f %.4f \t %.4f %.4f\n', mean(err_dslo1_text)/m*100,  std(err_dslo1_text)/m*100, mean(time_dslo1_text)/m*100, std(time_dslo1_text));
%fprintf(1,'dslo2_image \t %.4f %.4f \t %.4f %.4f\n', mean(err_dslo2_image)/m*100,  std(err_dslo2_image)/m*100, mean(time_dslo2_image)/m*100, std(time_dslo2_image));
%fprintf(1,'dslo2_text \t %.4f %.4f \t %.4f %.4f\n', mean(err_dslo2_text)/m*100,  std(err_dslo2_text)/m*100, mean(time_dslo2_text)/m*100, std(time_dslo2_text));
fprintf(1,'-------------------------------------------------------------------------------\n');

