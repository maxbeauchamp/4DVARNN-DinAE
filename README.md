# End-to-end learning of variational/energy models and solvers for inverse problems (especially with noisy and irregularly-sampled observations)

## Introduction

Associated preprints:
- Fixed-point solver: https://arxiv.org/abs/1910.00556
- Gradient-based solvers using automatic differentiation: https://arxiv.org/abs/2006.03653

License: CECILL-C license

Copyright IMT Atlantique/OceaniX, contributor(s) : R. Fablet, 21/03/2020

Contact person: ronan.fablet@imt-atlantique.fr
This software is a computer program whose purpose is to apply deep learning
schemes to dynamical systems and ocean remote sensing data.
This software is governed by the CeCILL-C license under French law and
abiding by the rules of distribution of free software.  You can  use,
modify and/ or redistribute the software under the terms of the CeCILL-C
license as circulated by CEA, CNRS and INRIA at the following URL
"http://www.cecill.info".
As a counterpart to the access to the source code and  rights to copy,
modify and redistribute granted by the license, users are provided only
with a limited warranty  and the software's author,  the holder of the
economic rights,  and the successive licensors  have only  limited
liability.
In this respect, the user's attention is drawn to the risks associated
with loading,  using,  modifying and/or developing or reproducing the
software by the user in light of its specific status of free software,
that may mean  that it is complicated to manipulate,  and  that  also
therefore means  that it is reserved for developers  and  experienced
professionals having in-depth computer knowledge. Users are therefore
encouraged to load and test the software's suitability as regards their
requirements in conditions enabling the security of their systems and/or
data to be ensured and,  more generally, to use and operate it in the
same conditions as regards security.
The fact that you are presently reading this means that you have had
knowledge of the CeCILL-C license and that you accept its terms.

## Architecture of the code

It will involve a joint learning scheme of operator <img src="https://render.githubusercontent.com/render/math?math=\Phi"> and solver <img src="https://render.githubusercontent.com/render/math?math=\Gamma"> through a bi-level optimization scheme:  
<img src="https://render.githubusercontent.com/render/math?math=\arg \min_{\Phi,\Gamma} \sum_n {\cal{L}} (x_n,\tilde{x}_n) \mbox{  s.t.  } \tilde{x}_n = \arg \min_x  U_\Phi \left ( x,y_n , \Omega_n\right)">  
where <img src="https://render.githubusercontent.com/render/math?math=U_\Phi"> is the 4DVAR cost function:  
<img src="https://render.githubusercontent.com/render/math?math=U_\Phi\left ( x , y , \Omega\right ) = \lambda_1 \sum_n \left \|x(t_n)-y(t_n)\right \|^2_{\Omega _{t_i}} %2B \lambda_2 \sum_n \left \|x(t_n) - \Phi(x)(t_n) \right \|^2">  
and <img src="https://render.githubusercontent.com/render/math?math=\cal{L}"> is the loss function to use when using automatic differential tools in the neural network. This loss is defined according to the GradType flag in utils/utils_solver/Compute_Grad.py:
  * GradType == 0: subgradient for prior <img src="https://render.githubusercontent.com/render/math?math=||x-g(x)||^2"> 
  * GradType == 1: true gradient using autograd for prior <img src="https://render.githubusercontent.com/render/math?math=||x-g(x)||^2">
  * GradType == 2: true gradient using autograd for prior <img src="https://render.githubusercontent.com/render/math?math=||x-g(x)||">
  * GradType == 3: true gradient using autograd for prior <img src="https://render.githubusercontent.com/render/math?math=||x-g1(x)||^2 %2B ||x-g2(x)||^2">
  * GradType == 4: true gradient using autograd for prior <img src="https://render.githubusercontent.com/render/math?math=||g(x)||^2">

According to the OptimType flag, the appropriate gradient-based solver <img src="https://render.githubusercontent.com/render/math?math=\Gamma"> is defined in the corresponding model_GradUpdateX.py for the minimization, i.e.:
  * if OptimType == 0, load utils/utils_solver/model_GradUpdate0.py : Gradient-based minimization using a fixed-step descent
  * if OptimType == 1, load utils/utils_solver/model_GradUpdate1.py : Gradient-based minimization using a CNN using a (sub)gradient as inputs
  * if OptimType == 2, load utils/utils_solver/model_GradUpdate2.py : Gradient-based minimization using a 2D convolutional LSTM using a (sub)gradient as inputs defined in ConvLSTM2d.py

The NN-based dynamical operator <img src="https://render.githubusercontent.com/render/math?math=\Phi"> is defined through flagAEtype:
  * utils/utils_nn/ConvAE.py: Convolutional auto-encoder
  * utils/utils_nn/GENN.py: Gibbs-Energy-based NN for using a 2d convolution ResNet architecture (utils/utils_nn/ResNetConv2d.py) and a 2d convolution Constrained layer with zero-weight at central point (utils/utils_nn/ConstrainedConv2d.py)

The general model for optimizing both parameters in <img src="https://render.githubusercontent.com/render/math?math=\Phi"> and <img src="https://render.githubusercontent.com/render/math?math=\Gamma"> is defined in utils/utils_solver/Model_4DVarNN_GradFP.py. It uses a set of (NProjection) fixed-point iterations to fill in the missing-data in the output of the NN; then, a predefined number (NGradIter) Gradient-based iterations are involved to converge through the best operator <img src="https://render.githubusercontent.com/render/math?math=\Phi"> and solver <img src="https://render.githubusercontent.com/render/math?math=\Gamma"> for the bi-level optimization scheme.

## Results

Below is an illustration of the results obtained on the daily velocity SSH field
when interpolating pseudo irregular and noisy observations (top-right panels) corresponding to 
along-track nadir (left) with additional pseudo wide-swath SWOT observations (right) built 
from an idealized groundtruth (top-left panels) with state-of-the-art optimal interpolation 
(bottom-left panels) and the newly proposed end-to-end learning approach (bottom-right panels): 

Nadir only                 |  Nadir+SWOT
:-------------------------:|:-------------------------:
![Farmers Market Finder Demo](figs/animation_grads_OSSE.gif)  |  ![Farmers Market Finder Demo](figs/animation_grads_OSSE.gif)


