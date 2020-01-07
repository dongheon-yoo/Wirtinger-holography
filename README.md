# Wirtinger holography
Follow up code for 
Chakravarthula, P., Peng, Y., Kollin, J., Fuchs, H., & Heide, F. (2019). 
"Wirtinger holography for near-eye displays," ACM Transactions on Graphics (TOG), 38(6), 213.

The work tries to optimize the phase profile displayed by phase-only spatial-light modulator for two-dimensional image reconstruction.
As the numerical reconstruction process from the phase profile to complex wavefront includes complex domain computations, the work utilizes Wirtinger derivatives which denote the gradient value of real functions with respect to complex variables.
With the gradient, the phase profile is updated by gradient descent algorithm.
Unlike the GS algorithm and double-phase encoding method, the work can achieve the hologram which can reconstruct complex wavefront with high image quality.

# Acknowledgements
Several function codes (square-grid, fft, ifft) in util borrows from [deepoptics] (https://github.com/vsitzmann/deepoptics). 
