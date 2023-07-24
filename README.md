<div align="center">
<h1>
  <br>
  <a href="http://acids.ircam.fr"><img src="images/logo_acids.png" alt="ACIDS" width="200"></a>
  <br>
  Deep machine learning in MaxMSP and Ableton Live
  <br>
</h1>

<h3>Deep machine Learning course for MaxMSP with tutorials in JAX, PyTorch and Numpy.</h3>


<center>
<i>Course given at the <a href="https://maxsummer2023.geidai.ac.jp/" target="_blank">MaxMSP summer school</a> (Tokyo university of the arts, Japan)</i><br/>
<b>Professor</b>: <a href="http://esling.github.io" target="_blank">Philippe Esling</a>
</center>

<h4>
  <a href="#lessons">Lessons</a> •
  <a href="#setup">Setup</a> •
  <a href="#details">Detailed lessons</a> •
  <a href="#contribution">Contribution</a> •
  <a href="#about">About</a>
</h4>
</div>

This repository contains the courses in creative machine learning applied to music and other creative mediums and how to develop and use these models inside MaxMSP and Ableton Live. 
This course is given at the [MaxMSP summer school](https://www.u-tokyo.ac.jp/en/index.html) ((Tokyo university of the arts, Japan)).
The courses slides along with a set of MaxMSP patches, Max4Live devices and interactive Jupyter Notebooks will be updated along the course to provide all the examples.
This course is proudly provided by the <a href="http://github.com/acids-ircam" target="_blank">ACIDS</a> group, part of the [Analysis / Synthesis](http://anasynth.ircam.fr/home/english) team at IRCAM.
This course can be followed entirely online through the set of [Google slides](http://slides.google.com) and [Colab](http://colab.google.com) notebooks links provided openly along each lesson.
However, we do recommend to fork the entire environment and follow the interactive notebooks through Jupyter lab to develop your
own coding environment.

**This course is meant to be only an introduction to all of these complex ideas.**
**We highly recommend to follow the companion "[Creative Machine Learning](http://www.github.com/acids-ircam/creative_ml)" course to truly acheive mastery of deep models development**.


<details>
  <summary>Table of Contents</summary>
  <ol>
    <li> <a href="#lessons">Lessons</a>
      <ul>
        <li><a href="#lessons">Machine learning</a></li>
        <li><a href="#lessons">Neural networks</a></li>
        <li><a href="#lessons">Advanced networks</a></li>
        <li><a href="#lessons">Deep learning</a></li>
        <li><a href="#lessons">Bayesian inference</a></li>
        <li><a href="#lessons">Latent models</a></li>
        <li><a href="#lessons">Approximate inference</a></li>
        <li><a href="#lessons">Variational auto-encoders and flows</a></li>
        <li><a href="#lessons">Generative adversarial networks</a></li>
        <li><a href="#lessons">Diffusion models</a></li>
      </ul>
    </li>
    <li> <a href="#details">Detailed lessons</a> </li>
    <li> <a href="#contribution">Contribution</a> </li>
    <li> <a href="#about">About</a> </li>
  </ol>
</details>


## Lessons

**Quick explanation.** For each of the following lessons, you will find a set of badges containing links to different parts of the course, which allows you to follow either the _online_ 
or _offline_ versions.

- Online:
[![Slides](https://img.shields.io/badge/Slides-online-7DA416.svg?style=flat-square&logo=googledrive)](https://docs.google.com/presentation/d/e/2PACX-1vRGy4H9JWjxK8d760O4pJT_7wfCett-rKjFV91d6jLkCHSMUntJjRA8a3r25M7_WrIDxggnjeXHdsi2/pub?start=false&loop=false&delayms=1000000&slide=id.p1) 
[![Colab](https://img.shields.io/badge/Notebook-colab-7DA416.svg?style=flat-square&logo=googlecolab)](https://colab.research.google.com/drive/1tAIsucXMqHJ0hVTYcuoUzBt69lSn86i0?usp=sharing) 
- Offline: 
[![Powerpoint](https://img.shields.io/badge/Slides-download-167DA4.svg?style=flat-square&logo=files)](00_introduction.pdf) 
[![Notebook](https://img.shields.io/badge/Notebook-download-167DA4.svg?style=flat-square&logo=jupyter)](00_setup.ipynb) 

Simply click on the corresponding badge to follow the lesson. Note that if the badge is displayed in red color as follows 
[![Slides](https://img.shields.io/badge/Slides-none-7D1616.svg?style=flat-square&logo=googledrive)]() 
it means that the content is not available yet and will be uploaded later. 

---

### [01 - Machine learning](01_machine_learning.pdf)

[![Slides](https://img.shields.io/badge/Slides-online-7DA416.svg?style=flat-square&logo=googledrive)](https://docs.google.com/presentation/d/e/2PACX-1vQiFd_QE7kW1OK2q4tFOtTtXXidXqrNNRDW6-sHp_KhqBa0j2dOvBTgyah-XULhDSSMwZIJvCy0SFQ8/pub?start=false&loop=false&delayms=10000) 
[![Powerpoint](https://img.shields.io/badge/Slides-download-167DA4.svg?style=flat-square&logo=files)](01_machine_learning.pdf) 
[![Colab](https://img.shields.io/badge/Notebook-colab-7DA416.svg?style=flat-square&logo=googlecolab)](https://colab.research.google.com/drive/1O38voJUpBlvynWJGt8m_gBUMq_LUeMO1?usp=sharing) 
[![Notebook](https://img.shields.io/badge/Notebook-download-167DA4.svg?style=flat-square&logo=jupyter)](01a_machine_learning.ipynb) 
    
    
This course provides a brief history of the development of artificial intelligence and introduces the general concepts of machine learning 
through a series of recent applications in the creative fields. This course also presents the pre-requisites, course specificities, toolboxes
and tutorials that will be covered and how to setup the overall environment.
This course introduces the formal notions required to understand machine learning along with classic problems of linear models 
for regression and classification. We discuss the mathematical derivation for optimization and various problems of overfitting, cross-validation
and model properties and complexity that are still quintessential in modern machine learning.

**Additional notebook on feature-based learning**

[![Colab](https://img.shields.io/badge/Notebook-none-7D1616.svg?style=flat-square&logo=googlecolab)]() 
[![Notebook](https://img.shields.io/badge/Notebook-none-7D1616.svg?style=flat-square&logo=jupyter)](01b_feature_based_learning.ipynb) 

---

### [02 - Neural networks](02_neural_networks.pdf)

[![Slides](https://img.shields.io/badge/Slides-online-7DA416.svg?style=flat-square&logo=googledrive)](https://docs.google.com/presentation/d/e/2PACX-1vRWEyOd3sR1T9Ruc4c3mBq4I3B80l7mS3wwZubWEUINYolJkWlyrNnfqRovw6a7Fbhw1a4xWggTuhZV/pub?start=false&loop=false&delayms=60000) 
[![Powerpoint](https://img.shields.io/badge/Slides-download-167DA4.svg?style=flat-square&logo=files)](02_neural_networks.pdf) 
[![Colab](https://img.shields.io/badge/Notebook-colab-7DA416.svg?style=flat-square&logo=googlecolab)](https://colab.research.google.com/drive/1PE2WFYL3fRao1JNQV9pnkAk2AbNc3z8y?usp=sharing) 
[![Notebook](https://img.shields.io/badge/Notebook-download-167DA4.svg?style=flat-square&logo=jupyter)](02_neural_networks.ipynb) 

This course provides a brief history of the development of neural networks along with all mathematical and implementation details. 
We discuss geometric perspectives on neurons and gradient descent and how these interpretation naturally extend to the case
of multi-layer perceptrons. Finally, we discuss the complete implementation of backpropagation through micro-grad.


---

### [03 - Advanced neural networks](03_advanced_networks.pdf)

[![Slides](https://img.shields.io/badge/Slides-online-7DA416.svg?style=flat-square&logo=googledrive)](https://docs.google.com/presentation/d/e/2PACX-1vTRtjopz1JOL1Tte1TyPL3QADCMdk2Wz39iHw_vWjeCR9fS_qtkoxgoF3jOBoECQVIqn0_RxzV9uBr2/pub?start=false&loop=false&delayms=60000) 
[![Powerpoint](https://img.shields.io/badge/Slides-download-167DA4.svg?style=flat-square&logo=files)](03_advanced_networks.pdf) 
[![Colab](https://img.shields.io/badge/Notebook-colab-7DA416.svg?style=flat-square&logo=googlecolab)](https://colab.research.google.com/drive/1OK7I0vHDwi-PNX3orxDKO_46v7WeDBBP?usp=sharing) 
[![Notebook](https://img.shields.io/badge/Notebook-download-167DA4.svg?style=flat-square&logo=jupyter)](03_advanced_networks.ipynb) 

In this course we introduce more advanced types of neural networks such as convolutional and recurrent architectures, along
with more advanced models (LSTM, GRU) and recent developments such as residual architectures.
We further discuss issues of regularization and initialization in networks.

---

### [04 - Deep learning](04_deep_learning.pdf)

[![Slides](https://img.shields.io/badge/Slides-online-7DA416.svg?style=flat-square&logo=googledrive)](https://docs.google.com/presentation/d/e/2PACX-1vTbg6NR7B92Db4fqcX1JWfW3eDhaR98OqHhv_OmuICo0q_TwbqQE_iD7wAvwl4HxH-IA1Ag3bLbKQWJ/pub?start=false&loop=false&delayms=60000) 
[![Powerpoint](https://img.shields.io/badge/Slides-download-167DA4.svg?style=flat-square&logo=files)](04_deep_learning.pdf) 
[![Colab](https://img.shields.io/badge/Notebook-colab-7DA416.svg?style=flat-square&logo=googlecolab)](https://colab.research.google.com/drive/1-OGz_vn-4gSa6jm5J6iXv_06X6dgKugq?usp=sharing) 
[![Notebook](https://img.shields.io/badge/Notebook-download-167DA4.svg?style=flat-square&logo=jupyter)](04_deep_learning.ipynb) 

We introduce here the fundamental shift towards deep learning, notably through the development of layerwise training 
and auto-encoders. We discuss how these are now less relevant through novel regularization methods and data availability.
We finish this course by discussing the recent attention mechanism and transformer architectures and
provide a set of modern applications.


---


## Setup

Along the tutorials, we provide a reference code for each section. 
This code contains helper functions that will alleviate you from the burden of data import and other sideline implementations. 
You will find designated spaces in each file to develop your solutions. 
The code is in Python (notebooks impending) and relies on the concept of [code sections](https://fr.mathworks.com/help/matlab/matlab_prog/run-sections-of-programs.html),
 which allows you to evaluate only part of the code (to avoid running long import tasks multiple times and concentrate on the question at hand.
 
**Please refer to the setup notebook to check if your configuration is correct**

### Dependencies

#### Python installation

In order to get the baseline scripts and notebooks to work, you need to have a working distribution of `Python 3.7` as a minimum (we also recommend to update your version to `Python 3.9`). We will also be using a large set of libraries, with the following ones being the most prohiminent

- [Numpy](https://numpy.org/)
- [Scikit-Learn](https://scikit-learn.org/)
- [PyTorch](https://pytorch.org/)
- [Jax](https://pytorch.org/)
- [Librosa](http://librosa.github.io/librosa/index.html)
- [Matplotlib](https://matplotlib.org/)

We highly recommend that you install [Pip](https://pypi.python.org/pypi/pip/) or [Anaconda](https://www.anaconda.com/download/) that will manage the automatic installation of those Python libraries (along with their dependencies). If you are using `Pip`, you can use the following commands

```
pip install -r requirements.txt
```

If you prefer to install all the libraries by hand to check their version, you can use individual commands

```
pip install numpy
pip install scikit-learn
pip install torch
pip install jax
pip install librosa
pip install matplotlib
```

For those of you who have never coded in Python, here are a few interesting resources to get started.

- [TutorialPoint](https://www.tutorialspoint.com/python/)
- [Programiz](https://www.programiz.com/python-programming)

#### Jupyter notebooks and lab

In order to ease following the exercises along with the course, we will be relying on [**Jupyter Notebooks**](https://jupyter.org/). If you have never used a notebook before, we recommend that you look at their website to understand the concept. Here we also provide the instructions to install **Jupyter Lab** which is a more integrative version of notebooks. You can install it on your computer as follows (if you use `pip`)

```
pip install jupyterlab
```

Then, once installed, you can go to the folder where you cloned this repository, and type in

```
jupyter lab
```

## Contribution

Please take a look at our [contributing](CONTRIBUTING.md) guidelines if you're interested in helping!

## About

Code and documentation copyright 2012-2023 by all members of ACIDS. 

Code released under the [CC-BY-NC-SA 4.0 licence](https://creativecommons.org/licenses/by-nc-sa/4.0/).
