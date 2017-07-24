function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));


z = X*theta;
temp_theta = theta;
temp_theta(1) = 0;
a = -sum(log(sigmoid(z)).*y);
b = -sum(log(ones(m,1)-sigmoid(z)).*(ones(m,1)-y));
J = (a+b)/m + (lambda/(2*m))*sum(temp_theta.^2);
t = sigmoid(z) - y;
grad = (1/m)*(t' * X)' + (lambda/m)*temp_theta;





% =============================================================


end
