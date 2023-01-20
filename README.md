# Experiment history

## Tiny datasets for experiment
```python
DATA_SPLIT = {
    'train': ['train/interlaken_00_c', 'train/zurich_city_01_e', 'train/zurich_city_02_c',
              'train/zurich_city_03_a', 'train/zurich_city_04_d', 'train/zurich_city_09_b', 'train/zurich_city_11_a',],
    'validation': ['train/interlaken_00_f', 'train/thun_00_a', 'train/zurich_city_05_a',
                   'train/zurich_city_07_a', 'train/zurich_city_10_b'],
    'trainval': ['train/interlaken_00_d',  'train/zurich_city_00_b', 'train/zurich_city_02_a', 'train/zurich_city_04_b',
                 'train/zurich_city_06_a', 'train/zurich_city_09_e'],
    'test': ['test/interlaken_00_a', 'test/interlaken_00_b', 'test/interlaken_01_a', 'test/thun_01_a',
             'test/thun_01_b', 'test/zurich_city_12_a', 'test/zurich_city_13_a', 'test/zurich_city_13_b',
             'test/zurich_city_14_a', 'test/zurich_city_14_b', 'test/zurich_city_14_c', 'test/zurich_city_15_a'],
}
```


<details>
<summary>0116</summary>

- Origin code

- GPU: A6000 48GB * 2 (Server)

- Batch size: 16

- Note: without validation code
</details>


<details>
<summary><b>0119</b></summary>

- Origin code

- GPU: A6000 48GB * 2 (Server)

- Batch size: 16

- **Train result:**

|Scene|Loss|EPE|1PE|2PE|RMSE|
|-----|----|---|---|---|----|
Final | Loss:  1.344 | EPE:  0.577 | 1PE: 12.269 | 2PE:  3.160 | RMSE:  1.275

- **Validation result:**

|Scene|Loss|EPE|1PE|2PE|RMSE|
|-----|----|---|---|---|----|
interlaken_00_f | Loss:  2.480 | EPE:  0.895 | 1PE: 20.700 | 2PE:  7.145 | RMSE:  2.060
thun_00_a | Loss:  3.036 | EPE:  1.019 | 1PE: 21.756 | 2PE:  7.173 | RMSE:  2.521
zurich_city_05_a | Loss:  2.405 | EPE:  0.860 | 1PE: 21.043 | 2PE:  6.825 | RMSE:  1.869
zurich_city_07_a | Loss:  1.486 | EPE:  0.637 | 1PE: 14.627 | 2PE:  2.963 | RMSE:  1.312
zurich_city_10_b | Loss:  2.325 | EPE:  0.842 | 1PE: 21.508 | 2PE:  7.359 | RMSE:  1.800
**Total** | **Loss:  2.231** | **EPE:  0.822** | **1PE: 19.873** | **2PE:  6.308** | **RMSE:  1.794**

</details>


<details>
  <summary>SE-CFF details</summary>

# SE-CFF 
### [S]tereo depth from [E]vents Cameras: [C]oncentrate and [F]ocus on the [F]uture
This is an official code repo for "**Stereo Depth from Events Cameras: Concentrate and Focus on the Future**"
CVPR 2022 [Yeong-oo Nam*](), [Mohammad Mostafavi*](https://smmmmi.github.io/), [Kuk-Jin Yoon](http://vi.kaist.ac.kr/project/kuk-jin-yoon/) and [Jonghyun Choi](http://ppolon.github.io/) (Corresponding author)

If you use any of this code, please cite both following publications:

```bibtex
@inproceedings{nam2022stereo,
  title     =  {Stereo Depth from Events Cameras: Concentrate and Focus on the Future},
  author    =  {Nam, Yeongwoo and Mostafavi, Mohammad and Yoon, Kuk-Jin and Choi, Jonghyun},
  booktitle =  {Proceedings of the IEEE/CVF Conference on Computer Vision and Patter Recognition},
  year      =  {2022}
}
```
```bibtex
@inproceedings{mostafavi2021event,
  title     =  {Event-Intensity Stereo: Estimating Depth by the Best of Both Worlds},
  author    =  {Mostafavi, Mohammad and Yoon, Kuk-Jin and Choi, Jonghyun},
  booktitle =  {Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages     =  {4258--4267},
  year      =  {2021}
}
```

### Maintainers
* [Yeong-oo Nam]()
* [Mohammad Mostafavi](https://smmmmi.github.io/)

## Table of contents
- [Pre-requisite](#pre-requisite)
    * [Hardware](#hardware)
    * [Software](#software)
    * [Dataset](#dataset)
- [Getting started](#getting-started)
- [Training](#training)
- [Inference](#inference)
    * [Pre-trained model](#pre-trained-model)
- [What is not ready yet](#what-is-not-ready-yet)
- [Benchmark website](#benchmark-website)
- [Related publications](#related-publications)
- [License](#license)

## Pre-requisite
The following sections list the requirements for training/evaluation the model.

### Hardware
Tested on:
- **CPU** - 2 x Intel(R) Xeon(R) Silver 4210R CPU @ 2.40GHz
- **RAM** - 256 GB
- **GPU** - 8 x NVIDIA A100 (40 GB)
- **SSD** - Samsung MZ7LH3T8 (3.5 TB)

### Software
Tested on:
- [Ubuntu 18.04](https://ubuntu.com/)
- [NVIDIA Driver 450](https://www.nvidia.com/Download/index.aspx)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)

### Dataset
Download [DSEC](https://dsec.ifi.uzh.ch/) datasets.

#### 📂 Data structure
Our folder structure is as follows:
```
DSEC
├── train
│   ├── interlaken_00_c
│   │   ├── calibration
│   │   │   ├── cam_to_cam.yaml
│   │   │   └── cam_to_lidar.yaml
│   │   ├── disparity
│   │   │   ├── event
│   │   │   │   ├── 000000.png
│   │   │   │   ├── ...
│   │   │   │   └── 000536.png
│   │   │   └── timestamps.txt
│   │   └── events
│   │       ├── left
│   │       │   ├── events.h5
│   │       │   └── rectify_map.h5
│   │       └── right
│   │           ├── events.h5
│   │           └── rectify_map.h5
│   ├── ...
│   └── zurich_city_11_c                # same structure as train/interlaken_00_c
└── test
    ├── interlaken_00_a
    │   ├── calibration
    │   │   ├── cam_to_cam.yaml
    │   │   └── cam_to_lidar.yaml
    │   ├── events
    │   │   ├── left
    │   │   │   ├── events.h5
    │   │   │   └── rectify_map.h5
    │   │   └── right
    │   │       ├── events.h5
    │   │       └── rectify_map.h5
    │   └── interlaken_00_a.csv
    ├── ...
    └── zurich_city_15_a                # same structure as test/interlaken_00_a
```

## Getting started

### Build docker image
```bash
git clone [repo_path]
cd event-stereo
docker build -t event-stereo ./
```

### Run docker container
```bash
docker run \
    -v <PATH/TO/REPOSITORY>:/root/code \
    -v <PATH/TO/DATA>:/root/data \
    -it --gpus=all --ipc=host \
    event-stereo
```

### Build deformable convolution
```bash
cd /root/code/src/components/models/deform_conv && bash build.sh
```

## Training
```bash
cd /root/code/scripts
bash distributed_main.sh
```

## Inference
```bash
cd /root/code
python3 inference.py \
    --data_root /root/data \
    --checkpoint_path <PATH/TO/CHECKPOINT.PTH> \
    --save_root <PATH/TO/SAVE/RESULTS>
```

### Pre-trained model
:gear: You can download pre-trained model from [here](https://drive.google.com/file/d/14_tmyMsXkd1H_0LWWe8GOXa_86OjsboG/view?usp=sharing)

## What is not ready yet
Some modules introduced in the paper are not ready yet. We will update it soon.
- Intensity image pre-processing code.
- E+I Model code.
- E+I train & test code.
- Future event distillation code.

## Benchmark website
The [DSEC website](https://dsec.ifi.uzh.ch) holds the benchmarks and competitions. 

:rocket: Our CVPR 2022 results (this repo), are available in the [DSEC website](https://dsec.ifi.uzh.ch/uzh/disparity-benchmark). We ranked better than the [state-of-the-art method from ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Mostafavi_Event-Intensity_Stereo_Estimating_Depth_by_the_Best_of_Both_Worlds_ICCV_2021_paper.pdf) 

:rocket: Our ICCV 2021 paper [Event-Intensity Stereo: Estimating Depth by the Best of Both Worlds](https://openaccess.thecvf.com/content/ICCV2021/papers/Mostafavi_Event-Intensity_Stereo_Estimating_Depth_by_the_Best_of_Both_Worlds_ICCV_2021_paper.pdf) ranked first in the [CVPR 2021 Competition](https://dsec.ifi.uzh.ch/cvpr-2021-competition-results) hosted by the [CVPR 2021 workshop on event-based vision](https://tub-rip.github.io/eventvision2021) and the [Youtube video](https://www.youtube.com/watch?v=xSidegLg0Ik&t=894s) from the competition.


## Related publications

- [Event-Intensity Stereo: Estimating Depth by the Best of Both Worlds - Openaccess ICCV 2021 (PDF)](https://openaccess.thecvf.com/content/ICCV2021/papers/Mostafavi_Event-Intensity_Stereo_Estimating_Depth_by_the_Best_of_Both_Worlds_ICCV_2021_paper.pdf)

- [E2SRI: Learning to Super Resolve Intensity Images from Events - TPAMI 2021 (Link)](https://www.computer.org/csdl/journal/tp/5555/01/09485034/1veokqDc14Q)

- [Learning to Reconstruct HDR Images from Events, with Applications to Depth and Flow Prediction - IJCV 2021](http://vi.kaist.ac.kr/wp-content/uploads/2021/04/Mostafavi2021_Article_LearningToReconstructHDRImages-1.pdf)

- [Learning to Super Resolve Intensity Images from Events - Openaccess CVPR 2020 (PDF)](https://openaccess.thecvf.com/content_CVPR_2020/papers/I._Learning_to_Super_Resolve_Intensity_Images_From_Events_CVPR_2020_paper.pdf)

- [Event-Based High Dynamic Range Image and Very High Frame Rate Video Generation Using Conditional Generative Adversarial Networks - Openaccess CVPR 2019 (PDF)](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Event-Based_High_Dynamic_Range_Image_and_Very_High_Frame_Rate_CVPR_2019_paper.pdf)


## License

MIT license.

</details>
