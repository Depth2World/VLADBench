# ✨✨VLADBench✨✨: Fine-Grained Evaluation of Large Vision-Language Models in Autonomous Driving 

<div align="center">

![VLADBench](https://img.shields.io/badge/Dataset-VLADBench-blue)
![VQA](https://img.shields.io/badge/Task-Autonomous--Driving--QA-red) 
![Multi-Modal](https://img.shields.io/badge/Modal-Image/Video-red)  
![Models](https://img.shields.io/badge/Models-Open--Source-green) 
![Models](https://img.shields.io/badge/Models-Closed--Source-green) 
![Models](https://img.shields.io/badge/Models-Domain--Specified-green) 

</div>
<div align="center" style="font-size: 22px; margin-top: 20px; margin-bottom: 20px; line-height: 2.5;">
  <a href="https://arxiv.org/pdf/2503.21505">📖 arXiv Paper</a> &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://huggingface.co/datasets/depth2world/VLADBench">🤗 Dataset</a>
</div>



# Overview

Current benchmarks designed for VLM-based AD face several notable limitations:

- **Coarse-grained Categories**: The underlying datasets of
the VLM-based models are often simplistic, typically categorizing tasks into perception, prediction, and planning
with reasoning, which are incomplete for evaluating the nuanced cognitive and reasoning abilities required for safe and
reliable AD. 

- **Lack of Dynamic Elements**: Both static and dynamic scenes are crucial for evaluating AD systems, a robust analysis of dynamic elements is particularly important for validating the temporal reasoning capabilities, especially in understanding traffic participant intentions within the scene and executing the nuanced spatio-temporal reasoning required for safe navigation. 

- **Homogeneous Data**:
Existing VLM-based AD datasets often suffer from a lack
of diversity, which limits the ability to test models across
a wide range of real-world scenarios. The narrow results
restrict the evaluation of zero-shot generalization and the
performance on challenging corner cases.

We introduce **VLADBench**, specifically designed to rigorously evaluate the capabilities of VLMs in AD. VLADBench employes a hierarchical structure that reflects the complex skill set required for reliable driving, progressing from fundamental scene and traffic elements comprehension to advanced reasoning and decision-making. 

- With 2000 static scenes and 3000 dynamic scenarios, VLADBench spans 5 primary domains: Traffic Knowledge Understanding (TKU), General Element Recognition (GER), Traffic Graph Generation (TGG), Target Attribute Comprehension (TAC), and Ego Decision-making and Planning(EDP). For a more detailed assessment, 11 secondary aspects and 29 tertiary tasks are defined, resulting in a total of 12K questions.
-  VLADBench is built from existing publicly available datasets, meticulously curated through a manual selection across 12 sources, aimed at challenging VLM capabilities in diverse challenging driving situations.
- To further investigate the intersections among the 5 key domains, we collect and construct approximately 1.4M AD-specific QAs from public resources. We then categorize these QAs
using GPT-4 and train models on individual domain-specific (DS) datasets. Finally, we validate the trained models on VLADBench to assess their performance across different domains.

<p align="center">
    <img src="./asset/sun.jpeg" width="50%" height="50%">
</p>



# Examples
<p align="center">
    <img src="./asset/domain_123.jpeg" width="90%" height="90%">
</p>
<p align="center">
    <img src="./asset/domain_45.jpeg" width="90%" height="90%">
</p>

# Evaluation Results
<p align="center">
    <img src="./asset/results_vlms.png" width="100%" height="100%">
</p>

For the detailed results of each tasks and the results from the large-scale models, please see the paper.


# Evaluation Pipeine

1. We provide a test example based on Qwen2-VL.[qwen2vl_all.py]
2. Run evaluate_vlm.py for scores.

**Note**  
**1. The bounding boxes in VLADBench are NOT resized. You should modify them in [prompt](https://github.com/Depth2World/VLADBench/blob/8156375d6b0e88ec5ca4f5a8119a9a0ee6c1ed18/qwen2vl_all.py#L79) and [evaluation](https://github.com/Depth2World/VLADBench/blob/8156375d6b0e88ec5ca4f5a8119a9a0ee6c1ed18/evaluate_utils.py#L170) for different VLMs.**  
**2. The finall scores do NOT inlucude the trajectory evaluation.**
