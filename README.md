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

### utils/utils_solver

* Model_4DVarNN_GradFP.py: First, it uses a set of FP iterations to define x_proj, the output of the NN. Then, it uses a set of Gradient-based iterations to find operator <img src="https://render.githubusercontent.com/render/math?math=\Phi"> minimizing the targeted 4DVAR cost:
<img src="https://render.githubusercontent.com/render/math?math=U_\Phi\left ( x , y , \Omega\right ) = \lambda_1 \sum_n \left \|x(t_n)-y(t_n)\right \|^2_{\Omega _{t_i}} %2B \lambda_2 \sum_n \left \|x(t_n) - \Phi(x)(t_n) \right \|^2">

It will involve a joint learning scheme of operator <img src="https://render.githubusercontent.com/render/math?math=\Phi"> and solver <img src="https://render.githubusercontent.com/render/math?math=\Gamma"> through a bi-level optimization scheme:
<img src="https://render.githubusercontent.com/render/math?math=\arg \min_{\Phi} \sum_n {\cal{L}} (x_n,\tilde{x}_n) \mbox{  s.t.  } \tilde{x}_n = \arg \min_x  U_\Phi \left ( x,y_n , \Omega_n\right)">
where <img src="https://render.githubusercontent.com/render/math?math=\cal{L}"> is loss function to use when using automatic differential tools. It is defined according to GradType in Compute_Grad.py:
  * GradType == 0: subgradient for prior ||x-g(x)||^2 
  * GradType == 1: true gradient using autograd for prior ||x-g(x)||^2
  * GradType == 2: true gradient using autograd for prior ||x-g(x)||
  * GradType == 3: true gradient using autograd for prior ||x-g1(x)||^2 + ||x-g2(x)||^2
  * GradType == 4: true gradient using autograd for prior ||g(x)||^2

According to OptimType, the class is initialized with the appropriate gradient-based solver <img src="https://render.githubusercontent.com/render/math?math=\Gamma"> in model_GradUpdateX.py for the minimization, i.e.:
  * if OptimType == 0, load model_GradUpdate0.py : Gradient-based minimization using a fixed-step descent
  * if OptimType == 1, load model_GradUpdate1.py : Gradient-based minimization using a CNN using a (sub)gradient as inputs
  * if OptimType == 2, load model_GradUpdate2.py : Gradient-based minimization using a 2D convolutional LSTM using a (sub)gradient as inputs defined in ConvLSTM2d.py

### utils/utils_nn

* ConvAE.py: Convolutional auto-encoder for operator Phi
* GENN.py: Gibbs-Energy-based NN for operator Phi
* ResNetConv2d.py: Define a ResNet architecture (Conv2d)   
* ConstrainedConv2d.py: Define a Constrained Conv2D Layer with zero-weight at central point

## Results

Below is an illustration of the results obtained on the daily velocity SSH field
when interpolating pseudo irregular and noisy observations (top-right panels) corresponding to 
along-track nadir (left) with additional pseudo wide-swath SWOT observations (right) built 
from an idealized groundtruth (top-left panels) with state-of-the-art optimal interpolation 
(bottom-left panels) and the newly proposed end-to-end learning approach (bottom-right panels): 

Nadir only                 |  Nadir+SWOT
:-------------------------:|:-------------------------:
![Farmers Market Finder Demo](figs/animation_grads_OSSE.gif)  |  ![Farmers Market Finder Demo](figs/animation_grads_OSSE.gif)


