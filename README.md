# Real-time Learning-based Motion Magnification at the Edge

This project is aimed at optimizing the learning-based video motion magnification (LB-VMM) model of [Oh et al. (2018)](https://arxiv.org/abs/1804.02684) in order to motion magnifiy live camera footage in real-time. The code of this project is written for NVIDIA Jetson Orin Nano Developer Kit hardware. The implemented optimized model is a modification of an existing PyTorch LB-VMM implementation by [cgst](https://github.com/cgst/motion-magnification). The optimized model uses [torch-tensorrt](https://pytorch.org/TensorRT/) acceleration. 

## Demo
[![Watch the video](https://img.youtube.com/vi/Vuabf5AByak/maxresdefault.jpg)](https://youtu.be/Vuabf5AByak)

The Torch-TensorRT optimized model can be found in the motion-magnification-master directory. This is a modified version of the repository of [cgst](https://github.com/cgst/motion-magnification).
