> [!WARNING]
> This is a pre-release software. It depends on ITensorOpenSystems.jl, which has not yet been publicly released.
>  As such, this repository contains code that will not work for users without access to ITensorOpenSystems.jl.


# OpenNetworks.jl
Quantum systems are becoming increasingly important for engineering and scientific purposes, finding applications
across quantum sensing, communications, and quantum computing. However, realistic quantum systems have dynamics 
which are dominated by noise: uncontrolled interactions between the quantum system and it's external environment.

Modelling the effects of noise on quantum systems is therefore an important and challenging problem. This repository
contains an emulator that approximately models the effects of noise on 2d quantum systems. The approach taken uses
tensor networks with loops, gauged by Belief Propagation. 

## Installation
To install the project, clone the repository:

```
git clone git@github.com:anthonypetertc/OpenNetworks.jl.git
```

Navigate into the cloned repository:

```
cd OpenNetworks.jl
```

Install the package using the package manager:

```julia
julia> using Pkg: Pkg

julia> Pkg.add(".")
```

## Tutorials
To see how this package can be used in examples see the tutorials found in `tutorials/`

