function  EOE(data_file)
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

addpath('./algorithms/firstorder', './algorithms/secondorder');

%load dataset
similarity_method = 'pearson';
load(sprintf('../data/textimage/original/%d', data_file));
load('../data/textimage/ID');
load(sprintf('../data/textimage/similarity/%s/%d', similarity_method, data_file));

sim_matrix = ctt';

m = size(ID, 2);

% data
data_Y = text_gnd(1:m,:);
data_Y = full(data_Y);
data_X = text_fea(1:m,:);
data_X = [data_X; co_text_fea];
[data_n, data_d] = size(data_X);

% set parameters
options.C = 5;
options.sigma = 4;
options.t_tick = round(size(ID,2)/10);
options.K = 10;
options.m = m;
options.eta = 0.005;

%%
% scale
MaxX=max(data_X,[],2);
MinX=min(data_X,[],2);
DifX=MaxX-MinX;
idx_DifNonZero=(DifX~=0);
DifX_2=ones(size(DifX));
DifX_2(idx_DifNonZero,:)=DifX(idx_DifNonZero,:);
data_X = bsxfun(@minus, data_X, MinX);
data_X = bsxfun(@rdivide, data_X , DifX_2);

% kernel
P = sum(data_X.*data_X,2);
P = full(P);
data_kernel = exp(-(repmat(P',data_n,1) + repmat(P,1,data_n)- 2*data_X*data_X')/(2*options.sigma^2));
%data_kernel = data_X * data_X';

kernel_or_not = 0;

vector_e = 2 .^ [-10:0];

for iter = 1 : length(vector_e)
	options.eta = vector_e(iter);

%% run experiments:
for i=1:size(ID,1),
   % fprintf(1,'running on the %d-th trial...\n',i);
    id_list = ID(i, :);

		%1. Perceptron
		switch kernel_or_not
			case 1
    [classifier, err_count, run_time, mistakes] = Perceptron(data_Y, data_kernel, options, id_list);
		  case 0
    [classifier, err_count, run_time, mistakes] = Perceptron_linear(data_Y, data_X, options, id_list);
		end
		error_Perceptron(i) = err_count;
		time_Perceptron(i) = run_time;
		mistakes_list_Perceptron(i,:) = mistakes;

		switch kernel_or_not
			case 1
    [classifier, err_count, run_time, mistakes] = Perceptron_OGD(data_Y, data_kernel, sim_matrix, options, id_list);
		  case 0
    [classifier, err_count, run_time, mistakes] = Perceptron_OGD_linear(data_Y, data_X, sim_matrix, options, id_list);
		end
		error_Perceptron_OGD(i) = err_count;
		time_Perceptron_OGD(i) = run_time;
		mistakes_list_Perceptron_OGD(i,:) = mistakes;

		%2. ALMA
    %[classifier, err_count, run_time, mistakes] = ALMA(data_Y, data_kernel, options, id_list);
    %[classifier, err_count, run_time, mistakes] = ALMA_linear(data_Y, data_X, options, id_list);
		%error_ALMA(i) = err_count;
		%time_ALMA(i) = run_time;
		%mistakes_list_ALMA(i,:) = mistakes;

		%3. ROMMA
    %[classifier, err_count, run_time, mistakes] = ROMMA(data_Y, data_kernel, options, id_list);
    %[classifier, err_count, run_time, mistakes] = ROMMA_linear(data_Y, data_X, options, id_list);
		%error_ROMMA(i) = err_count;
		%time_ROMMA(i) = run_time;
		%mistakes_list_ROMMA(i,:) = mistakes;

		%4. OGD
		switch kernel_or_not
			case 1
    [classifier, err_count, run_time, mistakes] = OGD(data_Y, data_kernel, options, id_list);
			case 0
    [classifier, err_count, run_time, mistakes] = OGD_linear(data_Y, data_X, options, id_list);
		end
		error_OGD(i) = err_count;
		time_OGD(i) = run_time;
		mistakes_list_OGD(i,:) = mistakes;

		switch kernel_or_not
			case 1
    [classifier, err_count, run_time, mistakes] = OGD_OGD(data_Y, data_kernel, sim_matrix, options, id_list);
		  case 0
    [classifier, err_count, run_time, mistakes] = OGD_OGD_linear(data_Y, data_X, sim_matrix, options, id_list);
		end
		error_OGD_OGD(i) = err_count;
		time_OGD_OGD(i) = run_time;
		mistakes_list_OGD_OGD(i,:) = mistakes;

		%5. PA
		switch kernel_or_not
			case 1
    [classifier, err_count, run_time, mistakes] = PA(data_Y, data_kernel, options, id_list);
		  case 0
    [classifier, err_count, run_time, mistakes] = PA_linear(data_Y, data_X, options, id_list);
		end
		error_PA(i) = err_count;
		time_PA(i) = run_time;
		mistakes_list_PA(i,:) = mistakes;
		
		switch kernel_or_not
			case 1
    [classifier, err_count, run_time, mistakes] = PA_OGD(data_Y, data_kernel, sim_matrix, options, id_list);
		  case 0
    [classifier, err_count, run_time, mistakes] = PA_OGD_linear(data_Y, data_X, sim_matrix, options, id_list);
		end
		error_PA_OGD(i) = err_count;
		time_PA_OGD(i) = run_time;
		mistakes_list_PA_OGD(i,:) = mistakes;

		%6. PA-I
		switch kernel_or_not
			case 1
    [classifier, err_count, run_time, mistakes] = PA_I(data_Y, data_kernel, options, id_list);
		  case 0
    [classifier, err_count, run_time, mistakes] = PA_I_linear(data_Y, data_X, options, id_list);
		end
		error_PA_I(i) = err_count;
		time_PA_I(i) = run_time;
		mistakes_list_PA_I(i,:) = mistakes;

		switch kernel_or_not
			case 1
    [classifier, err_count, run_time, mistakes] = PA_I_OGD(data_Y, data_kernel, sim_matrix, options, id_list);
		  case 0
    [classifier, err_count, run_time, mistakes] = PA_I_OGD_linear(data_Y, data_X, sim_matrix, options, id_list);
		end
		error_PA_I_OGD(i) = err_count;
		time_PA_I_OGD(i) = run_time;
		mistakes_list_PA_I_OGD(i,:) = mistakes;

		%7. PA-II
		switch kernel_or_not
			case 1
    [classifier, err_count, run_time, mistakes] = PA_II(data_Y, data_kernel, options, id_list);
		  case 0
    [classifier, err_count, run_time, mistakes] = PA_II_linear(data_Y, data_X, options, id_list);
		end
		error_PA_II(i) = err_count;
		time_PA_II(i) = run_time;
		mistakes_list_PA_II(i,:) = mistakes;

		switch kernel_or_not
			case 1
    [classifier, err_count, run_time, mistakes] = PA_II_OGD(data_Y, data_kernel, sim_matrix, options, id_list);
		  case 0
    [classifier, err_count, run_time, mistakes] = PA_II_OGD_linear(data_Y, data_X, sim_matrix, options, id_list);
		end
		error_PA_II_OGD(i) = err_count;
		time_PA_II_OGD(i) = run_time;
		mistakes_list_PA_II_OGD(i,:) = mistakes;

		%8. SOP
		%switch kernel_or_not
			%case 1
    %[classifier, err_count, run_time, mistakes] = PA_II(data_Y, data_kernel, options, id_list);
		  %case 0
    [classifier, err_count, run_time, mistakes] = SOP_linear(data_Y, data_X, options, id_list);
		%end
		error_SOP(i) = err_count;
		time_SOP(i) = run_time;
		mistakes_list_SOP(i,:) = mistakes;

		%switch kernel_or_not
			%case 1
    %[classifier, err_count, run_time, mistakes] = PA_II_OGD(data_Y, data_kernel, sim_matrix, options, id_list);
		  %case 0
    %[classifier, err_count, run_time, mistakes] = SOP_OGD_linear(data_Y, data_X, sim_matrix, options, id_list);
		%end
		%error_SOP_OGD(i) = err_count;
		%time_SOP_OGD(i) = run_time;
		%mistakes_list_SOP_OGD(i,:) = mistakes;

		%9. CW
		%switch kernel_or_not
			%case 1
    %[classifier, err_count, run_time, mistakes] = CW(data_Y, data_kernel, options, id_list);
		  %case 0
    [classifier, err_count, run_time, mistakes] = CW_linear(data_Y, data_X, options, id_list);
		%end
		error_CW(i) = err_count;
		time_CW(i) = run_time;
		mistakes_list_CW(i,:) = mistakes;

		%switch kernel_or_not
			%case 1
    %[classifier, err_count, run_time, mistakes] = CW_OGD(data_Y, data_kernel, sim_matrix, options, id_list);
		  %case 0
    %[classifier, err_count, run_time, mistakes] = CW_OGD_linear(data_Y, data_X, sim_matrix, options, id_list);
		%end
		%error_CW_OGD(i) = err_count;
		%time_CW_OGD(i) = run_time;
		%mistakes_list_CW_OGD(i,:) = mistakes;

		%10. SCW
		%switch kernel_or_not
			%case 1
    %[classifier, err_count, run_time, mistakes] = SCW(data_Y, data_kernel, options, id_list);
		  %case 0
    [classifier, err_count, run_time, mistakes] = SCW_linear(data_Y, data_X, options, id_list);
		%end
		error_SCW(i) = err_count;
		time_SCW(i) = run_time;
		mistakes_list_SCW(i,:) = mistakes;

		%switch kernel_or_not
			%case 1
    %[classifier, err_count, run_time, mistakes] = SCW_OGD(data_Y, data_kernel, sim_matrix, options, id_list);
		  %case 0
    %[classifier, err_count, run_time, mistakes] = SCW_OGD_linear(data_Y, data_X, sim_matrix, options, id_list);
		%end
		%error_SCW_OGD(i) = err_count;
		%time_SCW_OGD(i) = run_time;
		%mistakes_list_SCW_OGD(i,:) = mistakes;

		%11. SCW2
		%switch kernel_or_not
			%case 1
    %[classifier, err_count, run_time, mistakes] = SCW2(data_Y, data_kernel, options, id_list);
		  %case 0
    [classifier, err_count, run_time, mistakes] = SCW2_linear(data_Y, data_X, options, id_list);
		%end
		error_SCW2(i) = err_count;
		time_SCW2(i) = run_time;
		mistakes_list_SCW2(i,:) = mistakes;

		%switch kernel_or_not
			%case 1
    %[classifier, err_count, run_time, mistakes] = SCW2_OGD(data_Y, data_kernel, sim_matrix, options, id_list);
		  %case 0
    %[classifier, err_count, run_time, mistakes] = SCW2_OGD_linear(data_Y, data_X, sim_matrix, options, id_list);
		%end
		%error_SCW2_OGD(i) = err_count;
		%time_SCW2_OGD(i) = run_time;
		%mistakes_list_SCW2_OGD(i,:) = mistakes;

		%12. AROW
		%switch kernel_or_not
			%case 1
    %[classifier, err_count, run_time, mistakes] = AROW(data_Y, data_kernel, options, id_list);
		  %case 0
    [classifier, err_count, run_time, mistakes] = AROW_linear(data_Y, data_X, options, id_list);
		%end
		error_AROW(i) = err_count;
		time_AROW(i) = run_time;
		mistakes_list_AROW(i,:) = mistakes;

		%switch kernel_or_not
			%case 1
    %[classifier, err_count, run_time, mistakes] = AROW_OGD(data_Y, data_kernel, sim_matrix, options, id_list);
		  %case 0
    %[classifier, err_count, run_time, mistakes] = AROW_OGD_linear(data_Y, data_X, sim_matrix, options, id_list);
		%end
		%error_AROW_OGD(i) = err_count;
		%time_AROW_OGD(i) = run_time;
		%mistakes_list_AROW_OGD(i,:) = mistakes;

		%13. NAROW
		%switch kernel_or_not
			%case 1
    %[classifier, err_count, run_time, mistakes] = NAROW(data_Y, data_kernel, options, id_list);
		  %case 0
    [classifier, err_count, run_time, mistakes] = NAROW_linear(data_Y, data_X, options, id_list);
		%end
		error_NAROW(i) = err_count;
		time_NAROW(i) = run_time;
		mistakes_list_NAROW(i,:) = mistakes;

		%switch kernel_or_not
			%case 1
    %[classifier, err_count, run_time, mistakes] = NAROW_OGD(data_Y, data_kernel, sim_matrix, options, id_list);
		  %case 0
    %[classifier, err_count, run_time, mistakes] = NAROW_OGD_linear(data_Y, data_X, sim_matrix, options, id_list);
		%end
		%error_NAROW_OGD(i) = err_count;
		%time_NAROW_OGD(i) = run_time;
		%mistakes_list_NAROW_OGD(i,:) = mistakes;


		%14. NHERD
		%switch kernel_or_not
			%case 1
    %[classifier, err_count, run_time, mistakes] = NHERD(data_Y, data_kernel, options, id_list);
		  %case 0
    [classifier, err_count, run_time, mistakes] = NHERD_linear(data_Y, data_X, options, id_list);
		%end
		error_NHERD(i) = err_count;
		time_NHERD(i) = run_time;
		mistakes_list_NHERD(i,:) = mistakes;

		%switch kernel_or_not
			%case 1
    %[classifier, err_count, run_time, mistakes] = NHERD_OGD(data_Y, data_kernel, sim_matrix, options, id_list);
		  %case 0
    %[classifier, err_count, run_time, mistakes] = NHERD_OGD_linear(data_Y, data_X, sim_matrix, options, id_list);
		%end
		%error_NHERD_OGD(i) = err_count;
		%time_NHERD_OGD(i) = run_time;
		%mistakes_list_NHERD_OGD(i,:) = mistakes;


		%15. IELLIP
		%switch kernel_or_not
			%case 1
    %[classifier, err_count, run_time, mistakes] = IELLIP(data_Y, data_kernel, options, id_list);
		  %case 0
    [classifier, err_count, run_time, mistakes] = IELLIP_linear(data_Y, data_X, options, id_list);
		%end
		error_IELLIP(i) = err_count;
		time_IELLIP(i) = run_time;
		mistakes_list_IELLIP(i,:) = mistakes;

		%switch kernel_or_not
			%case 1
    %[classifier, err_count, run_time, mistakes] = IELLIP_OGD(data_Y, data_kernel, sim_matrix, options, id_list);
		  %case 0
    %[classifier, err_count, run_time, mistakes] = IELLIP_OGD_linear(data_Y, data_X, sim_matrix, options, id_list);
		%end
		%error_IELLIP_OGD(i) = err_count;
		%time_IELLIP_OGD(i) = run_time;
		%mistakes_list_IELLIP_OGD(i,:) = mistakes;


end


stat_file = sprintf('../stat/eoe/%d/%d-stat', iter, data_file);
save(stat_file, 'error_Perceptron', 'time_Perceptron', 'mistakes_list_Perceptron', 'error_OGD', 'time_OGD', 'mistakes_list_OGD', 'error_PA', 'time_PA', 'mistakes_list_PA', 'error_PA_I', 'time_PA_I', 'mistakes_list_PA_I', 'error_PA_II', 'time_PA_II', 'mistakes_list_PA_II', 'error_Perceptron_OGD', 'time_Perceptron_OGD', 'mistakes_list_Perceptron_OGD', 'error_OGD_OGD', 'time_OGD_OGD', 'mistakes_list_OGD_OGD', 'error_PA_OGD', 'time_PA_OGD', 'mistakes_list_PA_OGD', 'error_PA_I_OGD', 'time_PA_I_OGD', 'mistakes_list_PA_I_OGD', 'error_PA_II_OGD', 'time_PA_II_OGD', 'mistakes_list_PA_II_OGD', 'error_SOP', 'time_SOP', 'mistakes_list_SOP', 'error_CW', 'time_CW', 'mistakes_list_CW', 'error_AROW', 'time_AROW', 'mistakes_list_AROW', 'error_NAROW', 'time_NAROW', 'mistakes_list_NAROW', 'error_NHERD', 'time_NHERD', 'mistakes_list_NHERD', 'error_IELLIP', 'time_IELLIP', 'mistakes_list_IELLIP', 'error_SCW', 'time_SCW', 'mistakes_list_SCW', 'error_SCW2', 'time_SCW2', 'mistakes_list_SCW2');

fprintf(1,'-------------------------------------------------------------------------------\n');
fprintf(1,'                  number of mistakes,        cpu running time\n');
fprintf(1,'Perceptron \t %.4f %.4f \t %.4f %.4f\n', mean(error_Perceptron)/m*100,  std(error_Perceptron)/m*100, mean(time_Perceptron)/m*100, std(time_Perceptron));
fprintf(1,'Perceptron_OGD \t %.4f %.4f \t %.4f %.4f\n', mean(error_Perceptron_OGD)/m*100,  std(error_Perceptron_OGD)/m*100, mean(time_Perceptron_OGD)/m*100, std(time_Perceptron_OGD));
fprintf(1,'OGD \t %.4f %.4f \t %.4f %.4f\n', mean(error_OGD)/m*100,  std(error_OGD)/m*100, mean(time_OGD)/m*100, std(time_OGD));
fprintf(1,'OGD_OGD \t %.4f %.4f \t %.4f %.4f\n', mean(error_OGD_OGD)/m*100,  std(error_OGD_OGD)/m*100, mean(time_OGD_OGD)/m*100, std(time_OGD_OGD));
fprintf(1,'PA \t %.4f %.4f \t %.4f %.4f\n', mean(error_PA)/m*100,  std(error_PA)/m*100, mean(time_PA)/m*100, std(time_PA));
fprintf(1,'PA_OGD \t %.4f %.4f \t %.4f %.4f\n', mean(error_PA_OGD)/m*100,  std(error_PA_OGD)/m*100, mean(time_PA_OGD)/m*100, std(time_PA_OGD));
fprintf(1,'PA-I \t %.4f %.4f \t %.4f %.4f\n', mean(error_PA_I)/m*100,  std(error_PA_I)/m*100, mean(time_PA_I)/m*100, std(time_PA_I));
fprintf(1,'PA-I-OGD \t %.4f %.4f \t %.4f %.4f\n', mean(error_PA_I_OGD)/m*100,  std(error_PA_I_OGD)/m*100, mean(time_PA_I_OGD)/m*100, std(time_PA_I_OGD));
fprintf(1,'PA-II \t %.4f %.4f \t %.4f %.4f\n', mean(error_PA_II)/m*100,  std(error_PA_II)/m*100, mean(time_PA_II)/m*100, std(time_PA_II));
fprintf(1,'PA-II-OGD \t %.4f %.4f \t %.4f %.4f\n', mean(error_PA_II_OGD)/m*100,  std(error_PA_II_OGD)/m*100, mean(time_PA_II_OGD)/m*100, std(time_PA_II_OGD));
fprintf(1,'SOP \t %.4f %.4f \t %.4f %.4f\n', mean(error_SOP)/m*100,  std(error_SOP)/m*100, mean(time_SOP)/m*100, std(time_SOP));
fprintf(1,'CW \t %.4f %.4f \t %.4f %.4f\n', mean(error_CW)/m*100,  std(error_CW)/m*100, mean(time_CW)/m*100, std(time_CW));
fprintf(1,'AROW \t %.4f %.4f \t %.4f %.4f\n', mean(error_AROW)/m*100,  std(error_AROW)/m*100, mean(time_AROW)/m*100, std(time_AROW));
fprintf(1,'NAROW \t %.4f %.4f \t %.4f %.4f\n', mean(error_NAROW)/m*100,  std(error_NAROW)/m*100, mean(time_NAROW)/m*100, std(time_NAROW));
fprintf(1,'NHERD \t %.4f %.4f \t %.4f %.4f\n', mean(error_NHERD)/m*100,  std(error_NHERD)/m*100, mean(time_NHERD)/m*100, std(time_NHERD));
fprintf(1,'IELLIP \t %.4f %.4f \t %.4f %.4f\n', mean(error_IELLIP)/m*100,  std(error_IELLIP)/m*100, mean(time_IELLIP)/m*100, std(time_IELLIP));
fprintf(1,'SCW \t %.4f %.4f \t %.4f %.4f\n', mean(error_SCW)/m*100,  std(error_SCW)/m*100, mean(time_SCW)/m*100, std(time_SCW));
fprintf(1,'SCW2 \t %.4f %.4f \t %.4f %.4f\n', mean(error_SCW2)/m*100,  std(error_SCW2)/m*100, mean(time_SCW2)/m*100, std(time_SCW2));
fprintf(1,'-------------------------------------------------------------------------------\n');

%fprintf(1,'ALMA \t %.4f %.4f \t %.4f %.4f\n', mean(error_ALMA)/m*100,  std(error_ALMA)/m*100, mean(time_ALMA)/m*100, std(time_ALMA));
%fprintf(1,'ROMMA \t %.4f %.4f \t %.4f %.4f\n', mean(error_ROMMA)/m*100,  std(error_ROMMA)/m*100, mean(time_ROMMA)/m*100, std(time_ROMMA));

end
