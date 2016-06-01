# README #


### What is this repository for? ###
This repo contains a python package with tools for submodular maximization and learning mixtures of submodular functions (SSVM).

1. Inference: Lazy greedy maximization [Minoux. Optimization Techniques 1978], Cost-sensitive maximization [Leskovec et al. ACM SIGKDD 2007]
2. Learning: Mixture learning with SGD [Lin & Bilmes. UAI 2012] and AdaGrad [Duchi et al. J. of MLR 2011]

If you use this code, please cite: *Gygli, Grabner & Van Gool. Video Summarization by Learning Submodular Mixtures of Objectives. CVPR 2015*.  
[Paper](http://www.vision.ee.ethz.ch/~gyglim/vsum_struct/GygliCVPR15_vsum_struct.pdf) | [Bibtex](http://www.vision.ee.ethz.ch/~gyglim/vsum_struct/bibtex_cvpr2015.txt)

### How do I set it up? ###

* Requires: numpy, scipy
* Installation:
```
#!python
cd YourPath
git clone https://github.com/gyglim/gm_submodular.git
cd gm_submodular
python setup.py install --user

```


### Setting started ###
You can find an example [**here**](http://www.vision.ee.ethz.ch/~gyglim/gm_submodular/gm_submodular_usage.html).
This shows how to do submodular maximization as well as learning mixtures of objectives from training data.  
For more information check [[Lin & Bilmes. UAI 2012]](http://arxiv.org/pdf/1210.4871) and [[Gygli et al. CVPR 2015]](http://www.vision.ee.ethz.ch/~gyglim/vsum_struct/GygliCVPR15_vsum_struct.pdf)

### Licence (BSD) ###
Copyright (c) 2015, ETH Zurich. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
3. All advertising materials mentioning features or use of this software must display the following acknowledgement: This product includes software developed by the ETH Zurich
4. Neither the name of the ETH Zurich nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY ETH Zurich ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
