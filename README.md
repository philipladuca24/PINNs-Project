# PINNs
- M. Beekenkamp, A. Bhagavathula, P. LaDuca.

## Introduction
We are implementing a Physics Informed Neural Network, originally proposed in the paper “Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations” by Raissi, Perdikaris, and Karniadakis. Our idea is to create a partial derivative solver that is bound by physical laws, thus improving the accuracy, and minimizing the data requirements of our models. As three Physics concentrators, the motivation of this topic starts with our collective interest in the intersection between Physics and Deep Learning. We intend to apply the ideas brought by Karniadakis to a physical system they already explore, and then extend the method to different partial derivative systems such as magneto hydrodynamics or thermodynamics. This is a regression problem that involves minimizing/optimizing a loss function to output solutions to differential equations. 

## Original Paper
https://www.sciencedirect.com/science/article/pii/S0021999118307125 
