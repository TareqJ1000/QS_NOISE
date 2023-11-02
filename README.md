# QS_NOISE
Code in Python which numerically engineers processes to encode quantum noise from an arbitrary unitary. Specifically, the code deals with the engineering of the phase shift quantum noise, with fitness function (1 - (integral 1 + integral 2)), these two integrals being the only non-zero integral equations in our system of 16 equations. the .yaml file, which you can edit using notepad, allows you to change the number of coefficients, as well as the parameters for the genetic optimization (based on a package called PyGAD, so you'll need that first). The notebook and .py script both do the same thing in principle -- the latter enables you to parallelize the process on e.g. the supercomputer. 

To do: improve documentation in code/notebook, run preliminary rounds 

