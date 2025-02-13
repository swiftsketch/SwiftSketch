# SwiftSketch: A Diffusion Model for Image-to-Vector Sketch Generation

<p align="center">
<img src="docs/teaser.png" width="800px"/>
</p>

*SwiftSketch is a diffusion model that generates vector sketches by denoising a Gaussian in stroke coordinate space. It generalizes effectively across diverse classes and takes under a second to produce a single high-quality sketch.*


> Recent advancements in large vision-language models have enabled highly expressive and diverse vector sketch generation. However, state-of-the-art methods rely on a time-consuming optimization process involving repeated feedback from a pretrained model to determine stroke placement. Consequently, despite producing impressive sketches, these methods are limited in practical applications. In this work, we introduce SwiftSketch, a diffusion model for image-conditioned vector sketch generation that can produce high-quality sketches in less than a second. SwiftSketch operates by progressively denoising stroke control points sampled from a Gaussian distribution. Its transformer-decoder architecture is designed to effectively handle the discrete nature of vector representation and capture the inherent global dependencies between strokes. To train SwiftSketch, we construct a synthetic dataset of image-sketch pairs, addressing the limitations of existing sketch datasets, which are often created by non-artists and lack professional quality. For generating these synthetic sketches, we introduce ControlSketch, a method that enhances SDS-based techniques by incorporating precise spatial control through a depth-aware ControlNet. We demonstrate that SwiftSketch generalizes across diverse concepts, efficiently producing sketches that combine high fidelity with a natural and visually appealing style.

<a href="https://swiftsketch.github.io/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=red" height=20.5></a> 


# Description
Official implementation of our SwiftSketch paper.

# Code Coming Soon!
```
