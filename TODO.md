1. Does running circuits on less qubits make it faster?
2. Does running ITensorNetworks.jl on 260 qubits with bdim=100 make it slower?
3. GPU acceleration.
4. Write unit tests for edge cases of noisy circuit compilation.
5. Do a back of the envelope calculation of the how long it should take to do an svd and all the other basic operations that appear in the emulation. Try to work out if the slow down is fundamental or due to the way I have implemented it.
6. Think about IBM papers on noise models, and error mitigation. Try implementing this on my emulator and see what I can get to work. Could I run things on qiskit's device for free? https://docs.quantum.ibm.com/run/error-mitigation-explanation
7. Read paper that SC sends me on Transport properties of non-equilibrium open systems and see if I can implement this on my machine.
8. Make interface as simple and neat as possible.
9. Write some tutorials on how one can use this in the form of Pluto notebooks, and use this to make the interface easy to use.
10. As well as running circuits with well-defined noise models, also add: 1. method for specifying gates from mixing a unitary with an error (Lindblad) 2. A way for specifying evolution by Lindblad equation on the whole system and then Trotterizing this.


