function [classifier, error_count, run_time, mistakes] = Perceptron(Y, Kernel, options, id_list)
%--------------------------------------------------------------------------
t_tick = options.t_tick;

alpha = [];
SV = [];
ID = id_list;
error_count = 0;
mistakes = [];

% loop
tic

for t = 1 : length(ID)
	id = ID(t);
  y_t = Y(id);

	if(isempty(alpha))
		f_t = 0;
	else
		k_t = Kernel(id, SV(:))';
		f_t = alpha * k_t;
	end

	hat_y_t = sign(f_t);
	if(hat_y_t == 0)
		hat_y_t = 1;
	end

	l_t = (hat_y_t ~= y_t);
	if(l_t > 0)
		error_count = error_count + 1;
		alpha = [alpha y_t;];
		SV = [SV id];
	end

  if(mod(t, t_tick) == 0)
		mistakes = [mistakes error_count/t];
	end

end

classifier.SV = SV;
classifier.alpha = alpha;

run_time = toc;

