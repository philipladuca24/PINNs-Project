# Notes:

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

## Integrate burgers_preprocessing.py and visualiser.py into runner.py

- burgers_preprocessing.py complete, needs integration.
- visualiser.py structurally complete still needs to be debugged (see Debugging.md). Needs integration.
- Currently all of preprocessing runs in one go, as soon as the class is instantiated all the
variables are populated. Perhaps break into separate functions, and only call the portion that is 
needed as needed. 
- Perhaps functions in runner can be moved out? Move gradients out, move loss functions out, etc (like
how we did Beras) and only leave in the REPL (not built yet) the model runner, and the load/save methods
(not yet implemented).

## REPL

- Need to build REPL to allow toggling between retraining and loadine from terminal. 
- See HW5 or SearchEngine (from CSCI 0200). 

## Debug 

- Work through the debugging doc.
