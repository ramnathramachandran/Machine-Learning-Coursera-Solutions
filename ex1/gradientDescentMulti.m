function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
X = [ones(m, 1) X];
temp3=0;
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

theta1=0;


 for it=1:m
        
    init=X(it,:);
    init2=theta;
    %fprintf('size of X:%f\n',size(init));
    %fprintf('size of THETA:%f\n',size(init2));
    temp1=init .* init2;
    %fprintf('size of Temp1:%f\n',size(temp1));
    temp2=y(it);
    %fprintf('size of Temp1:%f\n',size(temp2));
    %theta1=theta1+(temp1-temp2);
    %theta2=theta2+(temp1-temp2)*X(it,2);
    temp3=temp1-temp2;
    
    
   end
    theta=temp3*X(it);
    %theta(1)=theta(1)-(alpha/m)*theta1;
    %theta(2)=theta(2)-(alpha/m)*theta2;
    theta=theta-(alpha/m)*theta;
    %fprintf('value of Theta:%f\n',theta);
    J_history(iter) = computeCostMulti(X, y, theta);











    % ============================================================

    % Save the cost J in every iteration   

end

end
