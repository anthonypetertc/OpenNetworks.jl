3. GPU acceleration.
4. Write unit tests for edge cases of noisy circuit compilation.
6. Think about IBM papers on noise models, and error mitigation. Try implementing this on my emulator and see what I can get to work. Could I run things on qiskit's device for free? https://docs.quantum.ibm.com/run/error-mitigation-explanation
7. Read paper that SC sends me on Transport properties of non-equilibrium open systems and see if I can implement this on my machine.
8. Make interface as simple and neat as possible.
9. Write some tutorials on how one can use this in the form of Pluto notebooks, and use this to make the interface easy to use.
10. As well as running circuits with well-defined noise models, also add: 1. method for specifying gates from mixing a unitary with an error (Lindblad) 2. A way for specifying evolution by Lindblad equation on the whole system and then Trotterizing this.
11. Read section "Avoid fields with abstract type." from julia docs, and think about how I can use parametrized types to improve the efficiency of my code.
12. Change the print out of the Channel object to make it more user friendly - like the ITensor one.
13. Make the circuit and VDMNetwork something that I can iterate over, and also think about what the print statement should be for these.

