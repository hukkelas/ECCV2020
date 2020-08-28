# ECCV 2020 Overview

## Useful Links

- [All papers](https://www.ecva.net/papers.php)

## Overview of Topics
- [Interesting Papers](#interesting-papers)
- [Image/Video Inpainting](#imagevideo-inpainting)
- [Generative Models](#generative-models)
- [Detection](#detection)

## ECCV Dailies
- [Monday](https://www.rsipvision.com/ECCV2020-Monday/)
- [Tuesday](https://www.rsipvision.com/ECCV2020-Tuesday/)
- [Wednesday](https://www.rsipvision.com/ECCV2020-Wednesday/)
- [Thursday](https://www.rsipvision.com/ECCV2020-Thursday/)

## Interesting papers:

- **Towards Streaming Perception** *(Best paper honorable mention)*
    * Proposes a simple quantitative metric for streaming data (e.g. video stream from autonomous systems) that works with several computer vision tasks.
    Examples for object detection and instance segmentation are shown.
     > The tradeoff between accuracy versus latency can now be measured quantitatively and there exists an optimal "sweet spot" that maximizes streaming accuracy, (2) asynchronous tracking and future forecasting naturally emerge as internal representations that enable streaming image understanding, and (3) dynamic scheduling can be used to overcome temporal aliasing, yielding the paradoxical result that latency is sometimes minimized by sitting idle and "doing nothing"."
     * [[ECCV Link]](https://papers.eccv2020.eu/paper/3553) [[Project Page]](https://www.cs.cmu.edu/~mengtial/proj/streaming/) [[Paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470460.pdf)
- **NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis** *(Best paper honorable mention)*
    * > We present a method that achieves state-of-the-art results for synthesizing novel views of complex scenes by optimizing an underlying continuous volumetric scene function using a sparse set of input views
    * Propose a simple fully-connected layer to map a 5D coordinate (spatial + direction - x, y, z, $\theta$, $\phi$) to the volume density and texture. Does not use voxelgrids to represent 3D data.
     * [[ECCV Link]](https://papers.eccv2020.eu/paper/1473/) [[Project Page]](https://www.matthewtancik.com/nerf) [[Paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460392.pdf)

- **RAFT: Recurrent All-Pairs Field Transforms for Optical Flow** *(Best paper)*
    * [[ECCV Link]](https://papers.eccv2020.eu/paper/3526/) [[Project Page]](https://github.com/princeton-vl/RAFT) [[Paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470392.pdf)
- **A Generalization of Otsu's Method and Minimum Error Thresholding**
    * A generalized histogram tresholding algorithm, which can be simplified to Otsu's method, Minimum Error Tresholding (MET), and weighted percentile thresholding.
    >  GHT thereby enables the continuous interpolation between those three algorithms, which allows thresholding accuracy to be improved significantly
    > GHT works by performing approximate maximum a posteriori estimation of a mixture of Gaussians with appropriate priors.
    * [[ECCV Link]](https://papers.eccv2020.eu/paper/3657/) [[Code]](https://github.com/jonbarron/hist_thresh) [[Paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700681.pdf) [[Video]](https://www.youtube.com/watch?v=rHtQQlQo1Q4)
- **A Metric Learning Reality Check** (poster)
    * > Deep metric learning papers from the past four years have consistently claimed great advances in accuracy, often more than doubling the performance of decade-old methods. In this paper, we take a closer look at the field to see if this is actually true. We find flaws in the experimental methodology of numerous metric learning papers, and show that the actual improvements over time have been marginal at best.
    * [[ECCV Link]](https://papers.eccv2020.eu/paper/5170/) [[Project Page]](https://github.com/jonbarron/hist_thresh) [[Paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700681.pdf) [[Blog]](https://medium.com/@tkm45/benchmarking-metric-learning-algorithms-the-right-way-90c073a83968)


## Image/Video Inpainting

- **Rethinking image inpainting via a mutual encoder-decoder with feature equalization** (oral)
    * Assumes that image inpainting consists of structure and texture generation, and propose a two-branch module.
    * [[ECCV Link]](https://papers.eccv2020.eu/paper/4179/) [[Project Page]](https://github.com/KumapowerLIU/Rethinking-Inpainting-MEDFE) [[Paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470715.pdf)

- **Hallucinating Visual Instances in Total Absentia**
    * > Unlike conventional image inpainting task that works on images with only part of a visual instance missing, HVITA concerns scenarios where an object is completely absent from the scene
    * [[ECCV Link]](https://papers.eccv2020.eu/paper/3120/) [[Paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123500256.pdf)


- **High-Resolution Image Inpainting with Iterative Confidence Feedback and Guided Upsampling**
    * Propose to predict the confidence for each pixel, then by basic thresholding they can re-inpaint corrupted regions of the image.
    * [[ECCV Link]](https://papers.eccv2020.eu/paper/3620/) [[Paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123660001.pdf)

- **DVI: Depth Guided Video Inpainting for Autonomous Driving**
    * Propose a method to remove moving objects. They do not consider the case when the vehicle is static?
    * [[ECCV Link]](https://papers.eccv2020.eu/paper/3120/) [[Paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123500256.pdf)


- **Guidance and Evaluation: Semantic-Aware Image Inpainting for Mixed Scenes** 
    * Propose different approaches to incorporate semantic segmentation for image inpainting.
    * [[ECCV Link]](https://papers.eccv2020.eu/paper/5897/) [[Paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123720681.pdf)


- **Proposal based Video Completion**
    * > we use 3D convolutions to obtain an initial inpainting estimate which is subsequently refined by fusing a generated set of proposals
    * [[ECCV Link]](https://papers.eccv2020.eu/paper/5605/) [[Paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123720035.pdf)


## Generative Models

- **Rewriting a Deep Generative Model** (oral)
    * Builds upon the assumption that a convolutional layer is an associative memory (key -> value mapping). With this, they can "rewrite" the  result of the generative model (e.g. make a rule that a horse always wears a hat).
    > To address the problem, we propose a formulation in which the desired rule is changed by manipulating a layer of a deep network as a linear associative memory.
    * [[ECCV Link]](https://papers.eccv2020.eu/paper/1326/) [[Project Page]](https://rewriting.csail.mit.edu/) [[Paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460341.pdf) [[Short Video]](https://www.youtube.com/watch?v=i2_-zNqtEPk)

- **Exploiting Deep Generative Prior for Versatile Image Restoration and Manipulation**
    * [[ECCV Link]](https://papers.eccv2020.eu/paper/3265/) [[Paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470256.pdf) [[code]](https://github.com/XingangPan/deep-generative-prior)


- **GAN Slimming: All-in-One GAN Compression by A Unified Optimization Framework**
    * [[ECCV Link]](https://papers.eccv2020.eu/paper/1488/) [[Paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490052.pdf)

- **Learning to Factorize and Relight a City**
    * Utilizing google street view they symthesise images at different lighnting conditions.
    * Learns disentanglement between static objects over time (buildings) vs dynamic objects.
    * [[ECCV Link]](https://papers.eccv2020.eu/paper/2473/) [[Paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490528.pdf)

- **ForkGAN: Seeing into the Rainy Night** (oral)
    * Propose an image translation network to translate images between weather conditions.
    > Our innovation is a fork-shape generator with one encoder and two decoders that disentangles the domain-specific and domain-invariant information.
    * [[ECCV Link]](https://papers.eccv2020.eu/paper/6147/) [[Project Page]](https://github.com/zhengziqiang/ForkGAN) [[Paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123480154.pdf)

- **High-Fidelity Synthesis with Disentangled Representation**
    * Utilizes VAE for their strong disentanglement, and a GAN-based generator for high-fidelity results.
    * [[ECCV Link]](https://papers.eccv2020.eu/paper/5320/) [[Paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123710154.pdf)

- **SRFlow: Learning the Super-Resolution Space with Normalizing Flow**
    * [[ECCV Link]](https://papers.eccv2020.eu/paper/4442/) [[Paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123500698.pdf)

- **What makes fake images detectable? Understanding properties that generalize**
    * > We seek to understand what properties of fake images make them detectable and identify what generalizes across different model architectures, datasets, and variations in training.
    * [[ECCV Link]](https://papers.eccv2020.eu/paper/5308/) [[Paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123710103.pdf)

## Detection
- **End-to-End Object Detection with Transformers** (oral)
    * Propose a method to view object detection as a direct set prediction problem by the use of transformers. This removes the need of hand-crafted modules, such as non-maxima suppression and anchor generation.
    * [[ECCV Link]](https://papers.eccv2020.eu/paper/832/) [[Project Page]](https://ai.facebook.com/blog/end-to-end-object-detection-with-transformers/) [[Paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460205.pdf) [[Code]](https://github.com/facebookresearch/detr)

- **Learning What to Learn for Video Object Segmentation** (oral)
    * Propose a model for tracking in video segmentation where the target object is given in the first frame
    > Our learner is designed to predict a powerful parametric model of the target by minimizing a segmentation error in the first frame. We further go beyond the standard few-shot learning paradigm by learning what our target model should learn in order to maximize segmentation accuracy.
    * [[ECCV Link]](https://papers.eccv2020.eu/paper/4440/) [[Project Page]](https://github.com/visionml/pytracking) [[Paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470766.pdf)

- **Du$^2$Net: Learning Depth Estimation from Dual-Cameras and Dual-Pixels** (oral)
    * > We present a novel approach based on neural networks for depth estimation that combines stereo from dual cameras with stereo from a dual-pixel sensor, which is increasingly common on consumer cameras
    * [[ECCV Link]](https://papers.eccv2020.eu/paper/2263/) [[Paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460562.pdf)

- **Conditional Convolutions for Instance Segmentation** (oral)
    * Propose to replace the RoIAlign/Crop in Mask R-CNN for class-conditional convolutions for instance segmentation.
    * [[ECCV Link]](https://papers.eccv2020.eu/paper/1105/) [[Paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460273.pdf)

- **Corner Proposal Network for Anchor-free, Two-stage Object Detection** (spotlight)
    * Propose a two-stage detector without anchors; the first stage detects corners, then all poosible combinations are considered to classify object detections in the second stage.
    * [[ECCV Link]](https://papers.eccv2020.eu/paper/492/) [[Paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123480409.pdf)

- **SimPose: Effectively Learning DensePose and Surface Normal of People from Simulated Data**
    * [[https://papers.eccv2020.eu/paper/6637/]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123740222.pdf)


## Other

- **Learning Object Placement by Inpainting for Compositional Data Augmentation**
    * > We propose a self-learning framework that automatically generates the necessary training data without any manual labeling by detecting, cutting, and inpainting objects from an image
    * [[ECCV Link]](https://papers.eccv2020.eu/paper/1973/) [[Paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580562.pdf)


