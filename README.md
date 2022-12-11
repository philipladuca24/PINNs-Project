# PINNs
- M. Beekenkamp, A. Bhagavathula, P. LaDuca.

## Introduction
We are attempting to re-create a Physics Informed Neural Network, originally proposed in the paper “Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations” by Raissi, Perdikaris, and Karniadakis. Our idea is to create a partial derivative solver that is bound by physical laws, thus improving the accuracy of the solutions. We are motivated by this topic because we are all Physics concentrators and are interested in the intersection between Physics and Deep Learning. We intend to apply the ideas brought by Karniadakis to a one-dimensional differential equation and generate a solution by training a feed-forward neural network that approximates the solution function. Our idea is to minimize a customized loss function which is a sum of the residual loss of the structure of the solution as well as its expected boundary condition behavior. Extending upon this idea, we also hope to explore solution generation in a data-poor / no-data regime using self-adaptive learning. 

## Motivating Literature
https://www.sciencedirect.com/science/article/pii/S0021999118307125 
https://arxiv.org/abs/2009.04544 
