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
grad=0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
cost=zeros(size(y));
temp=0;
% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
X = [ones(m, 1) X];
%fprintf('\n dimensions of a1 unit %f\n',size(X));
a2=sigmoid(X*Theta1');
temp_a2=a2;
a2= [ones(size(a2,1),1) a2];
%fprintf('\n dimensions of a2 unit %f\n',size(a2));
htheta=sigmoid(a2*Theta2');
%fprintf('\n sum of htheta unit %f\n',sum(htheta));
%fprintf('\n sum of y unit %f\n',sum(y));
eye_matrix=eye(num_labels);
y_mat=eye_matrix(y,:);
for i=1:m
for it=1:num_labels
    temp=temp+((-y_mat(i,it))*log(htheta(i,it)))-((1-y_mat(i,it))*log(1-htheta(i,it)));
    %fprintf('\n sum of temp unit %f\n',sum(temp));
    %fprintf('\n dimensions of cost unit %f\n',size(cost));
end
end
temp1=Theta1;
temp2=Theta2;
temp1(:,1)=zeros(size(temp1,1),1);
temp2(:,1)=zeros(size(temp2,1),1);
power1=temp1.^2;
power2=temp2.^2;
sum_reg1=sum(power1(:));
sum_reg2=sum(power2(:));
temp3=sum_reg1+sum_reg2;
J=(temp/m)+(lambda*temp3)/(2*m);
%+((lambda/(2*m))*sum_reg);



% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
%fprintf('\n dimensions of theta1 unit %f\n',size(Theta1));
%fprintf('\n dimensions of theta2 unit %f\n',size(Theta2));
%fprintf('\n dimensions of htheta unit %f\n',size(htheta));
%fprintf('\n dimensions of a2 unit %f\n',size(a2));
%fprintf('\n dimensions of y unit %f\n',size(y_mat));
%fprintf('\n dimensions of del3 unit %f\n',size(del3));

del3=htheta-y_mat;
fprintf('\n dimensions of del3 unit %f\n',size(del3));
del2=(del3*(Theta2(:,2:end))).*sigmoidGradient(X*(Theta1'));
fprintf('\n dimensions of del2 unit %f\n',size(del2));
%fprintf('\n dimensions of sigmoid gradient unit %f\n',size(tem));
%.*(sigmoidGradient(a2*(Theta2')));
delta1=del2'*X;
fprintf('\n dimensions of delta1 unit %f\n',size(delta1));
delta2=del3'*a2;
fprintf('\n dimensions of delta2  unit %f\n',size(delta2));
Theta1_grad=delta1/m;
Theta2_grad=delta2/m;
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

temp_theta1=Theta1;
temp_theta2=Theta2;
temp_theta1(:,1)=0;
temp_theta2(:,1)=0;
Theta1_grad=Theta1_grad+(lambda/m)*temp_theta1;
Theta2_grad=Theta2_grad+(lambda/m)*temp_theta2;
















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
