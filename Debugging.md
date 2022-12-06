# Debugging:

## 5/12/22 - lambda_adapt_pinn -> optimise_lambda error:

- minimising lambda_b
  0%|                                                                                                                                                                                        | 0/20001 [00:00<?, ?it/s]
4 length of params
4 length of params
4 length of params
4 length of params
21 length of params
  0%|                                                                                                                                                                                        | 0/20001 [00:00<?, ?it/s]
Traceback (most recent call last):
  File "/Users/maximbeekenkamp/Desktop/Computer_Science/CSCI_1470/Final_Project/PINNs-Project/Code/lambda_adapt_pinn.py", line 307, in <module>
    opt_state_lambda = optimise_lambda(opt_params_net, x, nu, lambda_b, lambda_f)
  File "/Users/maximbeekenkamp/Desktop/Computer_Science/CSCI_1470/Final_Project/PINNs-Project/Code/lambda_adapt_pinn.py", line 262, in optimise_lambda
    lamb = minimize(fun=loss_b, x0=lambda_b, args=(lambda_b, ), method="BFGS")
  File "/Users/maximbeekenkamp/anaconda3/envs/PINNs/lib/python3.9/site-packages/jax/_src/scipy/optimize/minimize.py", line 103, in minimize
    results = minimize_bfgs(fun_with_args, x0, **options)
  File "/Users/maximbeekenkamp/anaconda3/envs/PINNs/lib/python3.9/site-packages/jax/_src/scipy/optimize/bfgs.py", line 100, in minimize_bfgs
    f_0, g_0 = jax.value_and_grad(fun)(x0)
  File "/Users/maximbeekenkamp/anaconda3/envs/PINNs/lib/python3.9/site-packages/jax/_src/scipy/optimize/minimize.py", line 100, in <lambda>
    fun_with_args = lambda x: fun(x, *args)
  File "/Users/maximbeekenkamp/Desktop/Computer_Science/CSCI_1470/Final_Project/PINNs-Project/Code/lambda_adapt_pinn.py", line 179, in loss_b
    loss_b = lambda_b * ((net_u(params, lb) - 1) ** 2 + (net_u(params, ub)) ** 2)
  File "/Users/maximbeekenkamp/Desktop/Computer_Science/CSCI_1470/Final_Project/PINNs-Project/Code/lambda_adapt_pinn.py", line 113, in net_u
    return predict(params, X)
  File "/Users/maximbeekenkamp/Desktop/Computer_Science/CSCI_1470/Final_Project/PINNs-Project/Code/lambda_adapt_pinn.py", line 93, in predict
    for w, b in params[:-1]:
TypeError: iteration over a 0-d array

- minimising lambda_f
  0%|                                                                                                                                                                                        | 0/20001 [00:00<?, ?it/s]
4 length of params
4 length of params
4 length of params
4 length of params
  0%|                                                                                                                                                                                        | 0/20001 [00:00<?, ?it/s]
Traceback (most recent call last):
  File "/Users/maximbeekenkamp/Desktop/Computer_Science/CSCI_1470/Final_Project/PINNs-Project/Code/lambda_adapt_pinn.py", line 307, in <module>
    opt_state_lambda = optimise_lambda(opt_params_net, x, nu, lambda_b, lambda_f)
  File "/Users/maximbeekenkamp/Desktop/Computer_Science/CSCI_1470/Final_Project/PINNs-Project/Code/lambda_adapt_pinn.py", line 263, in optimise_lambda
    lamf = minimize(fun=loss_f, x0=lambda_f, args=(lambda_f,), method="BFGS")
  File "/Users/maximbeekenkamp/anaconda3/envs/PINNs/lib/python3.9/site-packages/jax/_src/scipy/optimize/minimize.py", line 103, in minimize
    results = minimize_bfgs(fun_with_args, x0, **options)
  File "/Users/maximbeekenkamp/anaconda3/envs/PINNs/lib/python3.9/site-packages/jax/_src/scipy/optimize/bfgs.py", line 100, in minimize_bfgs
    f_0, g_0 = jax.value_and_grad(fun)(x0)
TypeError: grad requires real- or complex-valued inputs (input dtype that is a sub-dtype of np.inexact), but got int32. If you want to use Boolean- or integer-valued inputs, use vjp or set allow_int to True.


- Both work for 4 calls of the minimize function, and bug on the 5th. Seems like params is changing shape on the 5th for lambda_b, and for
lambda_f the arguments contain an integer somewhere. 
- Current implementation of minimize reflects you only pass in the arguments to optimize for, the other arguments are inferred.
- Solutions for lambda_b say to turn params into a list, but we're doing this on line 91, so bug comes from elsewhere.


