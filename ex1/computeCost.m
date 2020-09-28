function J = computeCost(X,y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y);
% You need to return the following variables correctly 
J=0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
for iter=1:m

    init=X(iter,:).';
    init2=theta.';
    temp1=init2 * init;
    temp2=y(iter,:);
    J=J+((temp1-temp2)^2)/(2*m);

end
% =========================================================================

end
