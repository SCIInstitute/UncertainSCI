# Case: alpha = 1.5260, beta = -0.9500, n = 0, we focues q = 0 in the iteration

vgrid = np.cos( np.linspace(np.pi, 0, M) ), ug = [0, 0.5, 1.0]

ugrid is computed by ugrid[:,q] = (vgrid + 1) / 2 * (ug[q+1] - ug[q]) + ug[q]

![ugrid](images/ugrid.png)

xgrid is computed by xgrid[:,q] = idistinv(ugrid[:,q])

![xgrid](images/xgrid.png)

temp is computed by xgrid[:,q]

![temp](images/temp.png)

* In python, only the first element is -1. even though the first 35 elements
displayed in temp is -1. And temp[1] = ... = temp[25] = -0.9999999999983953.
This may cause error when computing xgrid by idistinv.

* In matlab, the first 15 elements are -1.

temp = (temp - xgrid[0,q]) / (xgrid[-1,q] - xgrid[0,q])

In both python and matlab, xgrid[0,q] = -1. and xgrid[-1,q] = -0.999999461617520.
and thus xgrid[-1,q] - xgrid[0,q] = 5.383815753212673e-07 (python) or
5.383824802640547e-07 in matlab.

* In python, temp - (-1.) gives only the first elements zeros and multiply by
xgrid[-1,q] - xgrid[0,q] yields temp

![tempexppy](images/tempexppy.png)

* In matlab, temp has the first 15 elements zeros.

![tempexpmat](images/tempexpmat.png)

exponents = [[-19, 0], [0, 0.60411718]].  temp = temp * (1 + vgrid)**exponents[0,q].
In both python and matlab, (1 + vgrid)**exponents[0,q] gives

![powerpy](images/powerpy.png)

* In python, this will make temp a very big number.

![tempfinalpy](images/tempfinalpy.png)

* In matlab, this gives first 15 zeros.

![tempfinalmat](images/tempfinalmat.png)

