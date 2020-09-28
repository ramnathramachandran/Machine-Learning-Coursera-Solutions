function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y);
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

theta1=0;
theta2=0;



    % ============================================================

    % Save the cost J in every iteration
    
    
   for it=1:m
        
    init=X(it,:).';
    init2=theta.';
    temp1=init2 * init;
    temp2=y(it);
    theta1=theta1+(temp1-temp2);
    theta2=theta2+(temp1-temp2)*X(it,2);
    
   end
   
    theta(1)=theta(1)-(alpha/m)*theta1;
    theta(2)=theta(2)-(alpha/m)*theta2;
    J_history(iter) = computeCost(X, y, theta);

end

end
