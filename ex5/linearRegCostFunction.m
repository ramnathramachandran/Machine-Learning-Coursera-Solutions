function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
%fprintf('size of theta %f',size(theta));
%fprintf('size of X %f',size(X));
htheta=X*theta;
%fprintf('size of htheta %f',size(htheta));
l1=sum((htheta-y).^2)/(2*m);
temp=theta;
temp(1)=0;
l2=((lambda/(2*m))*(sum(temp.^2)));
J=l1+l2;


%fprintf('size of y %f',size(y));
x1=X(:,2:end);
%fprintf('size of x1 %f',size(x1));
%fprintf('size of t1 %f',size((htheta-y)*x1'));
t1=X'*(htheta-y);
%fprintf('size of t1 %f',size(t1));
t3=(t1)/m;
t4=t3+((lambda/m)*(temp));
%grad(1)=t3;
%grad(2)=t4;
grad=t4;










% =========================================================================

grad = grad(:);

end
