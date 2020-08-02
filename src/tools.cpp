#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
	//Calculating RMSE
	VectorXd rmse(4);
	rmse << 0, 0, 0, 0;

	//Estimation vector size control
	if (estimations.size() != ground_truth.size()){
		std::cout << "Invalid estimation vector size!" << std::endl;
		return rmse;
	}

	//Estimation vector control (whether it's null)
	if (estimations.size() == 0){
		std::cout << "Estimation vector is null!" << std::endl;
		return rmse;
	}


	//Accumulate squared residuals
	for (unsigned int i = 0; i < estimations.size(); ++i){
		VectorXd residual = estimations[i] - ground_truth[i];
		residual = residual.array()*residual.array();
		rmse += residual;
	}

	//Calculating the mean
	rmse = rmse / estimations.size();

	//Calculating the squared root
	rmse = rmse.array().sqrt();
  
	return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
	//Calculating Jacobian
	MatrixXd  Hj(3,4);

	//State parameters
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);

	//The equations for the J.matrix
	float c1 = px*px + py*py;
	float c2 = sqrt(c1);
	float c3 = c1*c2;
	
	//Division control (whether it's zero)
	if (c1 < 0.0001){
		std::cout << "Division by zero error in CalculateJacobian!" << std::endl;
		std::cout << "A very small number is added to pass the error! " <<std::endl;
		c1 += 0.001;
	}

	//Computing the J.matrix
	Hj << px/c2, py/c2, 0, 0,
		-(py/c1), (px/c1), 0,0,
		py*(vx*py-vy*px) / c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;

	return Hj;
}