- [x] docstrings
    - [x] UncertainSCI/gp/__init__.py
    - [x] UncertainSCI/gp/kernel.py
    - [x] UncertainSCI/gp/mean.py
- [x] break out vis
- [x] 2-d example notebook
- [ ] code review

# later?
- [ ] dyanmic/loss threshold training of hypers and sample coordinate
- [ ] fix in/exact jax dtype problems
- [ ] case per-channel output covariance kronecker structure kernel
- [ ] scalar-identity structure ComputedArray (see UncertainSCI/_linalg.py)
    - [ ] Want to combine UncertainSCI/_linalg.py with UncertainSCI/utils/linalg.py?
- [ ] Finish add_point method in gp
    - resolves GH#118 and GH#119
- [ ] curried/projected gaussian process
- [ ] higher-d example notebook
    - [ ] possible to plot with projection/curried GP helper/wrapper
