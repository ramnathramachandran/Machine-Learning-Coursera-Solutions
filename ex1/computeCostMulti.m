function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.


for iter=1:m

    init=X(iter,:).';
    init2=theta.';
    temp1=init2 .* init;
    fprintf('size of Temp1:%f\n',size(temp1));
    temp2=y(iter,:);
    %fprintf('size of Temp2:%f\n',size(temp2));
    temp3=temp1-temp2;
    J=J+(temp3^2)/(2*m);
    fprintf('the cost funx=ction is%f:',J);

end


% =========================================================================

end
