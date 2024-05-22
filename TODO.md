1. Re-write functions in core so that they do not require a reference state or density matrix, but only the siteinds (or underlying graph).
2. Write unit tests for edge cases of noisy circuit compilation.
3. Write custom JSON parser for circuits imported from qiskit.
4. Write function for compiling noisy circuits from native ITensor instead of from qiskit import.
5. Write tests for preparing gates with non-standard parameter behaviour (Rzz, Rxx, Ryy).