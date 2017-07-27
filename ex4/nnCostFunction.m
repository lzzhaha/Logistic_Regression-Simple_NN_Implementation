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
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


display("Theta1 size: ");
size(Theta1)

for i=1:m
  z2 = Theta1*([1,X(i,:)])' ;  
  a2 = sigmoid(z2); 
  z3 = Theta2*([1;a2]);
  a3 = sigmoid(z3);
  res =zeros(num_labels,1);
  res(y(i)) = 1;
  J+= -(res')*log(a3) - (ones(num_labels,1)-res)'*log(ones(num_labels,1)-a3); 
  
  %Backpropagation algorithm
  delta3 = a3 - res;
  temp_theta2 = Theta2(:);
  %drop the first column
  temp_theta2 = reshape(temp_theta2(num_labels+1:end),num_labels,hidden_layer_size);
  delta2 = (temp_theta2')*delta3.*sigmoidGradient(z2);
  a2 = [1;a2];
  a1 = [1,X(i,:)]';
  Theta1_grad =Theta1_grad + delta2*(a1') ;
  Theta2_grad =Theta2_grad + delta3*(a2') ;
endfor
J = J/m;

Theta1 = Theta1(:);
Theta2 = Theta2(:);
Theta1 = reshape(Theta1(hidden_layer_size+1:end),hidden_layer_size,input_layer_size);
Theta2 = reshape(Theta2(num_labels+1:end),num_labels,hidden_layer_size);
J = lambda*(sum(sum(Theta1.^2))+sum(sum(Theta2.^2)))/(2*m) + J;

%Regularization
Theta1 = [zeros(size(Theta1,1),1), Theta1];
Theta2 = [zeros(size(Theta2,1),1),Theta2];
Theta1_grad = Theta1_grad./m + (lambda/m)*Theta1;
Theta2_grad = Theta2_grad./m + (lambda/m)*Theta2; 












% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
