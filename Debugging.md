# Debugging:

## 5/12/22 - lambda_adapt_pinn -> optimise_net / optimise_lambda error:

Traceback (most recent call last):
  File "/Users/maximbeekenkamp/Desktop/Computer_Science/CSCI_1470/Final_Project/PINNs-Project/Code/lambda_adapt_pinn.py", line 335, in <module>
    u_pred, lb_list, lf_list = model.train(nIter, x)
  File "/Users/maximbeekenkamp/Desktop/Computer_Science/CSCI_1470/Final_Project/PINNs-Project/Code/lambda_adapt_pinn.py", line 301, in train
    opt_state = self.optimise_net(it, opt_state, X, self.lambda_b, self.lambda_f)
  File "/Users/maximbeekenkamp/Desktop/Computer_Science/CSCI_1470/Final_Project/PINNs-Project/Code/lambda_adapt_pinn.py", line 260, in optimise_net
    g = grad(self.loss)(params, X, self.nu, lambda_b, lambda_f)
TypeError: Argument '<__main__.LambdaAdaptPINN object at 0x139838eb0>' of type <class '__main__.LambdaAdaptPINN'> is not a valid JAX type.

Traceback (most recent call last):
  File "/Users/maximbeekenkamp/Desktop/Computer_Science/CSCI_1470/Final_Project/PINNs-Project/Code/lambda_adapt_pinn.py", line 335, in <module>
    u_pred, lb_list, lf_list = model.train(nIter, x)
  File "/Users/maximbeekenkamp/Desktop/Computer_Science/CSCI_1470/Final_Project/PINNs-Project/Code/lambda_adapt_pinn.py", line 304, in train
    opt_state_lambda = self.optimise_lambda(opt_params_net, X, self.nu, self.lambda_b, self.lambda_f)
TypeError: Argument '<__main__.LambdaAdaptPINN object at 0x127a2cd00>' of type <class '__main__.LambdaAdaptPINN'> is not a valid JAX type.

- Error from the tracer instances inside of our @jit methods. Something about calling grad or optimise is bugging the code.
- Try and implement the Deep Chem Solution: https://github.com/deepchem/deepchem/blob/master/deepchem/models/jax_models/pinns_model.py
- Might need to take the functionality outside somehow?

Note only niter is being regarded as static, somehow jax thinks nu (float = 0.001) is non-hashable. Might be indicative of bigger issue.

Helpful websites: 
- https://github.com/deepchem/deepchem/blob/master/deepchem/models/jax_models/pinns_model.py
- https://github.com/google/jax/issues/4416
- https://github.com/google/jax/issues/5609


