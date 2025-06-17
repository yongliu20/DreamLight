# DreamLight: Towards Harmonious and Consistent Image Relighting
Yong Liu*, Wenpeng Xiao*, Qianqian Wang, Junlin Chen, Shiyin Wang, Yitong Wang, Xinglong Wu, Yansong Tang 
(*equal contribution)


<a href='https://arxiv.org/abs/2312.04089'><img src='https://img.shields.io/badge/ArXiv-2312.04089-red'></a> 



---
## 📖 Method
<p align="center">
 <img src="imgs/teaser.png" width="88%">
 <img src="imgs/result1.png" width="88%">
</p>






### Tab of Content
- [Installation](#1)
- [Usage](#2)
- [Cite](#3)

<span id="1"></span>


If you find any bugs due to carelessness on our part in organizing the code, feel free to contact us and point that!

### Installation
  Please install the dependencies:
  ```
  pip install -r requirements.txt
  ```
  - Note that we have updated the corresponding code about light adapter in the code of diffusers, thus please utilie the diffusers in this repo to replace the original diffusers.

   


<span id="2"></span>

### Usage

- #### Pretrained Weight
  We have provided the pretrained model weights and the pretrained CLIP image encoder weights. Please download them from here.



#### Inference 
- Please use the following command to perform inference for single or a group of images:
  ```
  python test/test.py
  ```
- Note that you should change the 'xxx/xxx' in test.py to the path of your corresponding path.



<span id="3"></span>
### Cite 

If you find our work helpful, we'd appreciate it if you could cite our paper in your work.
