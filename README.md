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

### [01 - Creative machine learning](https://docs.google.com/presentation/d/e/2PACX-1vT5O2dGSyQT2CTwXHM1HVDsunzskacJ6LhJBUMUbLRRg4C34krOSzqf8y75YD19QQrWfvBJypKlHd1Z/pub?start=false&loop=false&delayms=60000)

[![Slides](https://img.shields.io/badge/Slides-online-7DA416.svg?style=flat-square&logo=googledrive)](https://docs.google.com/presentation/d/e/2PACX-1vT5O2dGSyQT2CTwXHM1HVDsunzskacJ6LhJBUMUbLRRg4C34krOSzqf8y75YD19QQrWfvBJypKlHd1Z/pub?start=false&loop=false&delayms=60000) 
[![Powerpoint](https://img.shields.io/badge/Slides-download-167DA4.svg?style=flat-square&logo=files)](https://nubo.ircam.fr/index.php/s/p9HEpNy5yYWtQyy) 
    
    
This course provides a brief history of the development of artificial intelligence and introduces the general concepts of machine learning 
through a series of recent applications in the creative fields. This course also presents the pre-requisites, course specificities, toolboxes
and tutorials that will be covered and how to setup the overall environment.
Finally, we introduce the formal notions required to understand machine learning along with classic problems of linear models 
for regression and classification. We discuss the mathematical derivation for optimization and various problems of overfitting, cross-validation
and model properties and complexity that are still quintessential in modern machine learning.
We finish with a quick roundup of how to use existing deep models on GitHub.


---

### [02 - Using and developing deep models](https://docs.google.com/presentation/d/e/2PACX-1vRC4PpmQEgr0c6M7REkh_--slZA7xYXjZiS1MBjUbyUBBq-u10dDyzPoLKxlNXma4dd_YQC-JlnXOrw/pub?start=false&loop=false&delayms=60000)

[![Slides](https://img.shields.io/badge/Slides-online-7DA416.svg?style=flat-square&logo=googledrive)](https://docs.google.com/presentation/d/e/2PACX-1vRC4PpmQEgr0c6M7REkh_--slZA7xYXjZiS1MBjUbyUBBq-u10dDyzPoLKxlNXma4dd_YQC-JlnXOrw/pub?start=false&loop=false&delayms=60000) 
[![Powerpoint](https://img.shields.io/badge/Slides-download-167DA4.svg?style=flat-square&logo=files)](https://nubo.ircam.fr/index.php/s/3nsrd7TM7yzL8RL) 

This course provides a brief history of the development of neural networks along with all mathematical and implementation details. 
We discuss geometric perspectives on neurons and gradient descent and how these interpretation naturally extend to the case
of multi-layer perceptrons. We further introduce more advanced types of neural networks such as convolutional and recurrent architectures, along
with more advanced models (LSTM, GRU) and recent developments such as residual architectures.
We further discuss issues of regularization and initialization in networks.
Finally, we finish this course by discussing the recent attention mechanism and transformer architectures and
provide a set of modern applications.

---

### [03 - Embedding deep models in Max4Live](03_embedding_maxmsp.pdf)

[![Slides](https://img.shields.io/badge/Slides-online-7DA416.svg?style=flat-square&logo=googledrive)](https://docs.google.com/presentation/d/e/2PACX-1vSVhSnx_dazxBS-PuLaG5Xz9DMC7IT5VM4t3xBhxgrtjjKiDJ8lZntirwDuiIfuhhB6OhYIdnwG5Ucn/pub?start=false&loop=false&delayms=3000) 
[![Powerpoint](https://img.shields.io/badge/Slides-download-167DA4.svg?style=flat-square&logo=files)](03_advanced_networks.pdf) 

In this course, we will discuss how to embed existing deep models into MaxMSP. 
In order to capitalize on existing body of work in ML research, we will seek to run the model from Python (as most research is being done in this language).
Hence, we will first discuss OSC communication and how to setup an OSC protocol and exchange between MaxMSP and Python.
We briefly introduce the basics of Max4Live devices and show how to use these through the case study of the [FlowSynth](https://github.com/acids-ircam/flow_synthesizer) AI-based synthesizer control.

---

### [04 - Developing AI-based externals](04_deep_learning.pdf)

[![Slides](https://img.shields.io/badge/Slides-online-7DA416.svg?style=flat-square&logo=googledrive)](https://docs.google.com/presentation/d/e/2PACX-1vRMJDveCOO0EkjLTf_x8usNAOcvVaGq6VdwXy3d6GKCnXFgJumYb2r27qp7RmFCmnr9P8NFxRHWnqF9/pub?start=false&loop=false&delayms=3000) 
[![Powerpoint](https://img.shields.io/badge/Slides-download-167DA4.svg?style=flat-square&logo=files)](04_deep_learning.pdf) 

In this course, we go further in how we can potentially embed existing deep models into MaxMSP. 
One of the key problem of the previous method (OSC and Python) resides in the lack of computational efficiency.
Hence, we seek here to learn how to develop our own externals. We provide an introduction to the Max SDK in order to develop our own externals in C.
We further discuss the Torchscript and Tensorflow Lite interfaces that allow to embed Python models into efficient C codes.
We show how to use these skills through the case study of the [RAVE](https://github.com/acids-ircam/rave) real-time deep audio generation.

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
