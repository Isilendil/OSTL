function calculate_pearson( data_file, affinity_file)
%
%Description :  calculate_affinity calculate the affinity (transition probability matrixes 
%Parameters:
%           data_file       - data file name
%           affinity_file   - affinity file name


 disp('Calculating affinity matrix...');
    data = load(data_file);
    image_fea = data.image_fea;
    text_fea = data.text_fea;
    co_image_fea = data.co_image_fea;
    co_text_fea = data.co_text_fea;

    text_num = size(text_fea,1);
    image_num = size(image_fea,1);
    co_image_num = size(co_image_fea,1);
    cii_sim = zeros( co_image_num,image_num );
    co_text_num = size(co_text_fea,1);
    ctt_sim = zeros( co_text_num,text_num );

    sigma = 0.2;
    %text to text
    disp('Calculating text to text transition matrix...');
     
    P_tt = zeros(text_num, text_num);
    for idx1=1:(text_num-1)
        for idx2=(idx1+1):text_num
            fea1 = text_fea( idx1, : );
            fea2 = text_fea( idx2, : );
						x = fea1-mean(fea1);
						y = fea2-mean(fea2);
						sim = (x*y') / (sum(x.^2)*sum(y.^2));
            P_tt(idx1,idx2) = sim;
        end
    end
    %symmetric matrix
    P_tt=P_tt+P_tt';
    % diagonal line elements
    for idx1=1:text_num
        fea1 = text_fea( idx1, : );
        fea2 = text_fea( idx1, : );
				x = fea1-mean(fea1);
				y = fea2-mean(fea2);
				sim = (x*y') / (sum(x.^2)*sum(y.^2));
        P_tt(idx1,idx1) = sim;
    end

    %  image to image
     disp('Calculating image to image transition matrix...');
    P_ii = zeros(image_num, image_num);
    for idx1=1:(image_num-1)
        for idx2=(idx1+1):image_num
            fea1 = image_fea( idx1, : );
            fea2 = image_fea( idx2, : );
				    x = fea1-mean(fea1);
				    y = fea2-mean(fea2);
				    sim = (x*y') / (sum(x.^2)*sum(y.^2));
            P_ii(idx1,idx2) = sim;
        end
    end
    % symmetric matrix
    P_ii=P_ii+P_ii';
    % diagonal line elements
    for idx1=1:image_num
        fea1 = image_fea( idx1, : );
        fea2 = image_fea( idx1, : );
				x = fea1-mean(fea1);
				y = fea2-mean(fea2);
				sim = (x*y') / (sum(x.^2)*sum(y.^2));
        P_ii(idx1,idx1) = sim;
    end

    %  co-image to image
    disp('Calculating image to text transition matrix...');
    for idx1=1:co_image_num
        for idx2=1:image_num
            fea1 = co_image_fea( idx1, : );
            fea2 = image_fea( idx2, : );
				    x = fea1-mean(fea1);
				    y = fea2-mean(fea2);
				    sim = (x*y') / (sum(x.^2)*sum(y.^2));
            cii_sim(idx1,idx2) = sim;
        end
    end
    for idx1=1:co_text_num
        for idx2=1:text_num
            fea1 = co_text_fea( idx1, : );
            fea2 = text_fea( idx2, : );
				    x = fea1-mean(fea1);
				    y = fea2-mean(fea2);
				    sim = (x*y') / (sum(x.^2)*sum(y.^2));
            ctt_sim(idx1,idx2) = sim;
        end
    end
    % text to image
    %P_ti = text_fea*co_text_fea'*cii_sim;
		%why not use gaussian kernel function?
    P_ti = ctt_sim'*cii_sim;
    ctt = ctt_sim;
    cii = cii_sim;
    save( affinity_file, 'P_tt', 'P_ii', 'P_ti', 'ctt', 'cii');
end

