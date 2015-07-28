function calculate_pearson( data_file, affinity_file)
%
%Description :  calculate_affinity calculate the affinity (transition probability matrixes 
%Parameters:
%           data_file       - data file name
%           affinity_file   - affinity file name

data_file = '../original/en-ch';
affinity_file = '../similarity/pearson/en-ch';

 disp('Calculating affinity matrix...');
    data = load(data_file);
    CH_fea = data.CH_fea;
    EN_fea = data.EN_fea;
    co_CH_fea = data.co_CH_fea;
    co_EN_fea = data.co_EN_fea;

    EN_num = size(EN_fea,1);
    CH_num = size(CH_fea,1);
    co_CH_num = size(co_CH_fea,1);
    ccc_sim = zeros( co_CH_num,CH_num );
    co_EN_num = size(co_EN_fea,1);
    cee_sim = zeros( co_EN_num,EN_num );

    sigma = 0.2;
    %EN to EN
    disp('Calculating EN to EN transition matrix...');
     
    P_ee = zeros(EN_num, EN_num);
    for idx1=1:(EN_num-1)
        for idx2=(idx1+1):EN_num
            fea1 = EN_fea( idx1, : );
            fea2 = EN_fea( idx2, : );
						x = fea1-mean(fea1);
						y = fea2-mean(fea2);
						sim = (x*y') / (sum(x.^2)*sum(y.^2));
            P_ee(idx1,idx2) = sim;
        end
    end
    %symmetric matrix
    P_ee=P_ee+P_ee';
    % diagonal line elements
    for idx1=1:EN_num
        fea1 = EN_fea( idx1, : );
        fea2 = EN_fea( idx1, : );
				x = fea1-mean(fea1);
				y = fea2-mean(fea2);
				sim = (x*y') / (sum(x.^2)*sum(y.^2));
        P_ee(idx1,idx1) = sim;
    end

    %  CH to CH
     disp('Calculating CH to CH transition matrix...');
    P_cc = zeros(CH_num, CH_num);
    for idx1=1:(CH_num-1)
        for idx2=(idx1+1):CH_num
            fea1 = CH_fea( idx1, : );
            fea2 = CH_fea( idx2, : );
				    x = fea1-mean(fea1);
				    y = fea2-mean(fea2);
				    sim = (x*y') / (sum(x.^2)*sum(y.^2));
            P_cc(idx1,idx2) = sim;
        end
    end
    % symmetric matrix
    P_cc=P_cc+P_cc';
    % diagonal line elements
    for idx1=1:CH_num
        fea1 = CH_fea( idx1, : );
        fea2 = CH_fea( idx1, : );
				x = fea1-mean(fea1);
				y = fea2-mean(fea2);
				sim = (x*y') / (sum(x.^2)*sum(y.^2));
        P_cc(idx1,idx1) = sim;
    end

    %  co-CH to CH
    disp('Calculating CH to EN transition matrix...');
    for idx1=1:co_CH_num
        for idx2=1:CH_num
            fea1 = co_CH_fea( idx1, : );
            fea2 = CH_fea( idx2, : );
				    x = fea1-mean(fea1);
				    y = fea2-mean(fea2);
				    sim = (x*y') / (sum(x.^2)*sum(y.^2));
            ccc_sim(idx1,idx2) = sim;
        end
    end
    for idx1=1:co_EN_num
        for idx2=1:EN_num
            fea1 = co_EN_fea( idx1, : );
            fea2 = EN_fea( idx2, : );
				    x = fea1-mean(fea1);
				    y = fea2-mean(fea2);
				    sim = (x*y') / (sum(x.^2)*sum(y.^2));
            cee_sim(idx1,idx2) = sim;
        end
    end
    % EN to CH
    %P_ec = EN_fea*co_EN_fea'*ccc_sim;
		%why not use gaussian kernel function?
    P_ec = cee_sim'*ccc_sim;
    cee = cee_sim;
    ccc = ccc_sim;
    save( affinity_file, 'P_ee', 'P_cc', 'P_ec', 'cee', 'ccc');
end

