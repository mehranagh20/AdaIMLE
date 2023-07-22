---
title: "Adaptive IMLE"
subtitle: "Adaptive IMLE for Few-shot Pretraining-free Generative Modelling"
---

<div class="container-fluid text-center">
    <div class="row content">
        <div class="col-sm-12">
        <!-- add names next to each other seperated by , -->
            <p class="name"><a href="https://mehranagh20.github.io/">Mehran Aghabozorgi</a>, <a href="https://sites.google.com/view/niopeng/home">Shichong Peng</a>, <a href="https://www.math.ias.edu/~ke.li/">Ke Li</a></p>
            <p class="affiliation">APEX Lab, Simon Fraser University</p>    
        </div>
    </div>
</div>

<!-- add three buttons for Paper, Code and Video -->
<div class="container-fluid text-center">
    <div class="row content">
        <div class="col-sm-12">
            <li class="list-inline-item btn-shadow">
                <a href="https://github.com/mehranagh20/AdaIMLE" title="Github">
                    <span class="fa-stack fa-lg" aria-hidden="true">
                        <i class="fas fa-circle fa-stack-2x"></i>
                        <i class="fa fa-code fa-stack-1x fa-inverse"></i>
                    </span>
                </a>
            </li>
            <!-- add link to paper https://openreview.net/pdf?id=CNq0JvrDfw -->
            <li class="list-inline-item">
                <a href="https://openreview.net/pdf?id=CNq0JvrDfw" title="Paper">
                    <span class="fa-stack fa-lg" aria-hidden="true">
                        <i class="fas fa-circle fa-stack-2x"></i>
                        <i class="fa fa-file-pdf fa-stack-1x fa-inverse"></i>
                    </span>
                </a>
            </li>
            <li class="list-inline-item">
                <a href="https://www.youtube.com/watch?v=Z3Z3XZ3Z3Z3" title="Video">
                    <span class="fa-stack fa-lg" aria-hidden="true">
                        <i class="fas fa-circle fa-stack-2x"></i>
                        <i class="fa fa-video fa-stack-1x fa-inverse"></i>
                    </span>
                </a>
            </li>
        </div>
    </div>

</div>

<!-- add divider in markdown -->
<!-- --- -->

## Overview

<div class="div-test">
We present Adaptive IMLE, a generative modeling approach that covers all the modes and produces high-quality results. Adaptive IMLE is capble of learning from a few samples from scratch without any auxiliary datasets. 
</div>

<div class="div-test">
We apply our method to the challenging task of few-shot unconditional image generation with as few as 100 data examples.
<!-- include cenetred image-->
</div>

<img src="assets/img/pretraining-free.png" alt="pretraining-free setting" width="100%" height="100%">

---

### Results

<div class="div-test">
Adaptive IMLE generates high-quality images:
</div>
<img src="assets/img/unconditional.jpg" width="80%" class="center">
<div class="div-test">
Smooth and meaningful interpolation:
</div>
<!-- six rows of ceneted images -->
<img src="assets/img/interp-3.jpg" width="70%" class="center">
<img src="assets/img/interp-5.jpg" width="70%" class="center">
<img src="assets/img/interp-9.jpg" width="70%" class="center">
<!-- add a bit of spacing -->
<br/>
<img src="assets/img/obama-interp-1.jpg" width="70%" class="center">
<img src="assets/img/obama-interp-6.jpg" width="70%" class="center">
<img src="assets/img/obama-interp-2.jpg" width="70%" class="center">

---

<div class="div-test">
Adaptive IMLE trains fast and stably!
</div>
<img src="assets/img/100-shot-obama-fig-fid.gif" width="50%" class="center">

---

<div class="div-test">
Adaptive IMLE is capable of learning from even fewer samples without any pretraining or auxiliary datasets!

The following figure shows the results of training Adaptive IMLE on 10-shot datasets.

</div>
<img src="assets/img/10-shot2.jpg" width="80%" class="center">

---

<div id="citation">
    <h2>Citation</h2>
        <pre  class="citation"><code>@inproceedings{aghabozorgi2023adaimle,
title={Adaptive IMLE for Few-shot Pretraining-free Generative Modelling
},
author={Mehran Aghabozorgi and Shichong Peng and Ke Li},
booktitle={International Conference on Machine Learning},
year={2023}
}</code></pre>
        <div>
