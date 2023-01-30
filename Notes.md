# Notes:

## Point-adaptive PINNs

- take points and output loss to create dict (return loss as list instead of average), order dict based on loss, take top x percent and add points between them and points on left and right, for edge points only add to left or right depending on edge, 
- which optimization to run between each new point creation?
- for percent lowest loss remove the points and move adjacent points to be evenly spaced in interval or just remove point?
- stop optimization at a loss threshold

## Self-adaptive PINNs

- L-BFGS-B optimizer not working within jit.

## No data PINNs

- Resources: https://github.com/madagra/basic-pinn/blob/main/burgers_equation_1d.py, https://towardsdatascience.com/solving-differential-equations-with-neural-networks-afdcf7b8bcc4

## Load and Save Models 

- Need to find way to load and save models for Jax, current models take ~12min to train. 
Potential solution: https://github.com/google/flax/discussions/1876 Would need to be able to 
toggle between retraining and loading at will. Considering we aren't using flax, we might
want to save /load models via pickle instead: 
https://stackoverflow.com/questions/64550792/how-do-i-save-an-optimizer-state-of-jax-trained-model
- Jax doesn't have native support for loading and saving models, so we will need to either use
other libraries or we could save the weights and biases in a list format and then write a script 
in which we initialize the model with the saved weights and biases.  

- Will need to separate the model builder from the visualiser in this doc.
- Will try and immitate the code from HW5 -> toggle between retraining and loading in
the terminal. Might need to implement a REPL.
- Need to restructure code into classes to allow this. Current code slightly unreadable.
- Incomplete burgers_model.py revamp.
- runner.py hasn't been started.

## Generate docstrings!

- Need to build documentation for our functions so that we can all use our code and that its easy for
people to continue coding and / or debug.

## Debug 

- Work through the debugging doc.
