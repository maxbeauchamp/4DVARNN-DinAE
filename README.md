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

## Results

Below is an illustration of the results obtained when interpolating pseudo 
irregular and noisy wide-swath SWOT observations built (top-right panel) from 
an idealized groundtruth (top-left panel) with state-of-the-art optimal interpolation
(bottom-left panel) and the new proposed end-to-end learning approach: 

![Farmers Market Finder Demo](figs/animation_grads_OSSE.gif)
