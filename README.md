# README #


### What is this repository for? ###
This repo contains a python package with tools for submodular maximization and learning mixtures of submodular functions (SSVM).

1. Inference: Lazy greedy maximization [Minoux. Optimization Techniques 1978], Cost-sensitive maximization [Leskovec et al. ACM SIGKDD 2007]
2. Learning: Mixture learing with SGD [Lin & Bilmes. UAI 2012] and AdaGrad [Duchi et al. J. of MLR 2011]

### How do I get set up? ###

* Requires: numpy, scipy
* git clone https://bitbucket.org/gyglim/gm_submodular
* python setup.py --user

### Setting started ###
* See http://www.vision.ee.ethz.ch/~gyglim/gm_submodular/gm_submodular_usage.html

### Licence ###
Copyright (c) 2015, ETH Zurich. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
3. All advertising materials mentioning features or use of this software must display the following acknowledgement: This product includes software developed by the ETH Zurich
4. Neither the name of the ETH Zurich nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY ETH Zurich ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.