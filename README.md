# Bottom-up conditioned top-down pose estimation (BUCTD) 

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rethinking-pose-estimation-in-crowds/pose-estimation-on-crowdpose)](https://paperswithcode.com/sota/pose-estimation-on-crowdpose?p=rethinking-pose-estimation-in-crowds)

This repository contains the official code for our paper: [Rethinking pose estimation in crowds: overcoming the detection information-bottleneck and ambiguity](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhou_Rethinking_Pose_Estimation_in_Crowds_Overcoming_the_Detection_Information_Bottleneck_ICCV_2023_paper.pdf). 
[[YouTube Video]](https://www.youtube.com/watch?v=BHZnA-CZeZY) [[Website]](https://amathislab.github.io/BUCTD/)


- Sep 2023: We released the code :)
- July 2023: This work is accepted to ICCV 2023 🎉
- June 2023: BUCTD was also presented at the 2023 [CV4Animals workshop at CVPR](https://www.cv4animals.com)
- June 2023: An earlier version can be found on [arxiv](https://arxiv.org/abs/2306.07879)

<img src="media/BUCTD_fig1.png" width="600">

- This code will also be integrated in [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut)!

### Installation 

We developed and tested our models with ```python=3.8.10, pytorch=1.8.0, cuda=11.1```. Other versions may also be suitable.

<details>
  <summary>Instructions</summary>
   
   1. Clone this repo, and in the following we will call the directory that you cloned ${BUCTD_ROOT}.

   ```sh
   git clone https://github.com/amathislab/BUCTD.git
   cd ${BUCTD_ROOT}
   ```

   2. Install Pytorch and torchvision

   Follow the instructions on https://pytorch.org/get-started/locally/.
   ```sh
   # an example:
   conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
   ```

   3. Install additional dependencies
   
   ```sh
   pip install -r requirements.txt
   ```

   4. Install [COCOAPI](https://github.com/cocodataset/cocoapi)
   
   ```sh
   # COCOAPI=/path/to/clone/cocoapi
   git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
   cd $COCOAPI/PythonAPI
   # Install into global site-packages
   make install
   # Alternatively, if you do not have permissions or prefer
   # not to install the COCO API into global site-packages
   python setup.py install --user
   ```
   
   5. Install [CrowdPoseAPI](https://github.com/Jeff-sjtu/CrowdPose) exactly in the same way as COCOAPI.

   6. Install NMS
   
   ```sh
   cd ${BUCTD_ROOT}/lib
   make
   ```
   
</details>


### Training
<details>
  <summary>Instructions</summary>

***Generative sampling***

You can use the script: ```train_BUCTD_synthesis_noise.sh```.

***Empirical sampling***

You can match your own bottom-up (BU) models by updating the scripts in [./data_preprocessing/](./data_preprocessing/). 

If you do not want to match your own BU models for training, we provide the training annotations. You can download the annotations [here](https://zenodo.org/records/10039883).

During inference, we use different BU/one-stage model's predictions (e.g. PETR, CID) as Conditions. The result files can be downloaded from the link above. 

</details>

### Testing

We also provide the best model per human dataset along with the testing scripts.</summary>
  
### COCO

| Model | Sampling strategy | Image Size | Condition | AP | Weights | Script |
|-------|---------------|------------|-----------|----|----------|------
|  BUCTD-preNet-W48     |        Generative sampling       |    384x288        |     PETR     |  77.8  |          [download](https://zenodo.org/records/10039883/files/COCO-BUCTD-preNet-W48.pth?download=1)     | [script](./scripts/test/test_BUCTD_prenet_gen_sample.sh)  |


### OCHuman

| Model | Sampling strategy | Image Size | Condition | AP_val | AP_test | Weights | Script |
|-------|---------------|------------|-----------|----|--------|----------|------|
|  BUCTD-CoAM-W48     |        Generative sampling (3x iterative refinement)      |    384x288        |     CID-W32      |  49.0  |    48.5  |    [download](https://zenodo.org/records/10039883/files/COCO-BUCTD-CoAM-W48.pth?download=1)     | [script](./scripts/test/test_BUCTD_COAM_gen_sample.sh) |


### CrowdPose

| Model | Sampling strategy | Image Size | Condition | AP | Weights | Script |
|-------|---------------|------------|-----------|----|----------|------|
|  BUCTD-CoAM-W48     |        Generative sampling       |    384x288        |      PETR      |  78.5  |      [download](https://zenodo.org/records/10039883/files/CrowdPose-BUCTD-CoAM-W48.pth?download=1)     | [script](./scripts/train/train_BUCTD_COAM_gen_sample.sh)


### Code Acknowledgements

We are grateful to the authors of [HRNet](https://github.com/HRNet/deep-high-resolution-net.pytorch), [MIPNet](https://rawalkhirodkar.github.io/mipnet), and [TransPose](https://github.com/yangsenius/TransPose) as our code builds on their excellent work. 

## Reference

If you find this code or ideas presented in our work useful, please cite:

[Rethinking pose estimation in crowds: overcoming the detection information-bottleneck and ambiguity (ICCV)](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhou_Rethinking_Pose_Estimation_in_Crowds_Overcoming_the_Detection_Information_Bottleneck_ICCV_2023_paper.pdf) by Mu Zhou*, Lucas Stoffl*, Mackenzie W. Mathis and Alexander Mathis ([arxiv](https://arxiv.org/abs/2306.07879))


```
@InProceedings{Zhou_2023_ICCV,
    author    = {Zhou, Mu and Stoffl, Lucas and Mathis, Mackenzie Weygandt and Mathis, Alexander},
    title     = {Rethinking Pose Estimation in Crowds: Overcoming the Detection Information Bottleneck and Ambiguity},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {14689-14699}
}
```

# License

BUCTD is released under the Apache 2.0 license. Please see the [LICENSE](LICENSE) file for more information.
