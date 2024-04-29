### GPU-Accelerated Multi-Material Decomposition for Dual-Energy CT

####  Project Overview
This repository provides a PyTorch implementation of a GPU-accelerated algorithm for multi-material decomposition of dual-energy CT (DECT) images, replicating the method proposed in the 2013 paper by GE:
- "A Flexible Method for Multi-Material Decomposition of Dual-Energy CT Images" [https://ieeexplore.ieee.org/document/6600785](https://ieeexplore.ieee.org/document/6600785).

#### Credits and Acknowledgments

- This implementation is recomposed from the `multi-mat-decomp` project by gjadick available on GitHub: [https://github.com/gjadick/multi-mat-decomp](https://github.com/gjadick/multi-mat-decomp)
- This project also utilizes GitHub Copilot and ChatGPT-4 for code suggestions and debugging assistance.

#### Improvement
- Using a single NVIDIA GeForce RTX 3090 GPU and a 40-core Intel(R) Xeon(R) Silver 4210R CPU @ 2.40GHz, this implementation achieves a speed improvement of 800 to 1000 times compared to pixel-by-pixel decomposition implementations. Processing speed for thousands of DECT data sets has been improved from approximately 100s per image pair to 119ms per image pair.

#### Usage
You may run the decomposition algorithm by simply using the following command:

```bash
python gpu-MMD.py
```

#### Dataset 
- The code is designed to be compatible with any DECT Pair Dataset with certain spectrums.

#### Contact Information
For further information or inquiries, please contact:
- **Email:** medphyxhli@buaa.edu.cn
You may also refer to my related repository  `DualEnergyCTSynthesis
` [https://github.com/KryogenBlue/DualEnergyCTSynthesis](https://github.com/KryogenBlue/DualEnergyCTSynthesis) for more details.
