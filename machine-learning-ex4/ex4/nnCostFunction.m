function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));        %25x401
Theta2_grad = zeros(size(Theta2));        %10x26

% ====================== YOUR CODE HERE ======================

%%%h = sigmoid (X*theta)
%%%

%-----------forward propogation---------------
a1 = [ones(m, 1) X];  %5000x401    adding 1 to first activation layer. Ex [ones(2,1) zeros(2,3)] = 1 0 0 0 / 1 0 0 0 
z2 = a1*Theta1';                     %5000x25
h2 = sigmoid(z2);                    %5000x25
a2 = [ones(size(z2, 1), 1) h2];      %5000x26
z3 = a2*Theta2';                     %5000x26
h3 = sigmoid(z3);                    %5000x10
h=h3;

%-------------- Proper format of Training Y  -----------
k_output_patern = eye(num_labels);
y_binary_vector = zeros(m, num_labels);
for i=1:m
	y_int = y(i);
	y_binary_vector(i, :)= k_output_patern (y_int, :);  %converting y from int to vector[k]  ex, 0 0 0 0 1 0 0 0 
end

%-------------- regularization -----------
reg_sum = (sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2))  );  % Theta1(:,2:end) skips the biased node weight 
%--------------cost function for final layer-----------
J = 1/m * sum(sum( (-y_binary_vector).*log(h)-(1.-y_binary_vector).*log(1.-h))) + (lambda/(2*m))*reg_sum;  % cost with regularization

%------------------ back prop---------------
sigma3 = h .- y_binary_vector;   							 %5000x10
z2_biased = [ones(size(z2,1),1) z2];                         %5000x26  adding bias
sigma2 = (sigma3*Theta2) .* (sigmoidGradient(z2_biased));    %5000x26
sigma2 = sigma2 (:,2:end);                                   %5000x25  removing bias
delta2 = sigma3' * a2;                                       %10x26
delta1 = sigma2' * a1;                                       %25x401
Theta1_grad = delta1 ./ m;
Theta2_grad = delta2 ./ m;

%-----------------greadient regularization---------------------
Theta1_j0=[zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];     % Replacing first biased theta with zeroes
Theta2_j0=[zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];
reg1=(lambda/m) .* Theta1_j0;                               % regularization value to be added to grad
reg2=(lambda/m) .* Theta2_j0;

Theta1_grad = Theta1_grad + reg1;
Theta2_grad = Theta2_grad + reg2;

%------------------unrolled gradient-----------------------
grad = [Theta1_grad(:) ; Theta2_grad(:)];                    % unrolled both to single vector



end
