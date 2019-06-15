# Image-derain-via-CGAN
Use a CGAN to remove the rain in the image.

## Environment

1. Python 2.7

2. Tensorflow 1.12

## Get started

1. Clone !
```
git clone https://github.com/yoyotv/Image-derain-via-CGAN.git
```

2. Download the training and testing data from [here](https://drive.google.com/drive/folders/1qCHxRfTEPSm4ap90NOHOqhOcJShqP8dp). Unzip them and put under Image-derain-via-CGAN-master/code/.

3. Download the vgg19 pretrain model from [here](https://drive.google.com/drive/folders/1BpIqvXIq__0w6Y3hoOxxmpKFxPTj9htR). Then put it under Image-derain-via-CGAN-master/code/.

4. Create two empty folder named "model" and "tensorboard" in order to store the model and tensorboard files.

## Method

* Basically, we are doing the re-implementation in [1]. Although completely followed the steps described in [1], we did not get the claimed performance, so refer to [2], we apply vgg19 and using the higher level features described in [2].

## Train

1. Assume you are now under Image-derain-via-CGAN-master/code/

2. Open gan_main.py and modify the training dataset path.

3. Run 
```
python gan_main.py
```

## Deploy

1. Assume you are now under Image-derain-via-CGAN-master/code/

2. Open gan_main_deploy.py and modify the deploy image path and model that youwant to load.

3. Run 
```
python gan_main_deploy.py
```

## What we found

* Using Adam optimizer will always cause the discriminator too strong, so we apply SGD in discriminator. Also, as long as the learning rate smaller than 0.002, the discriminator always overcomes the generator.

* We tried different combinations between coefficients. It seems like the keypoint in this case is the relation between GAN,  ,VGG and Raw loss coefficient. e.g. Model 13 result is quite good (2019.06.15).

## Results

Some "OK" results we get.

| Input | Output | Ground truth |
|:-:|:-:|:-:|
| <img src="https://raw.githubusercontent.com/yoyotv/Image-derain-via-CGAN/master/figures/1_input.JPG" >|<img src="https://raw.githubusercontent.com/yoyotv/Image-derain-via-CGAN/master/figures/1_output.JPG" >|<img src="https://raw.githubusercontent.com/yoyotv/Image-derain-via-CGAN/master/figures/1_ground_truth.JPG" >|
| <img src="https://raw.githubusercontent.com/yoyotv/Image-derain-via-CGAN/master/figures/2_input.JPG" >|<img src="https://raw.githubusercontent.com/yoyotv/Image-derain-via-CGAN/master/figures/2_output.JPG" >|<img src="https://raw.githubusercontent.com/yoyotv/Image-derain-via-CGAN/master/figures/2_ground_truth.JPG" >|
| <img src="https://raw.githubusercontent.com/yoyotv/Image-derain-via-CGAN/master/figures/3_input.JPG" >|<img src="https://raw.githubusercontent.com/yoyotv/Image-derain-via-CGAN/master/figures/3_output.JPG" >|<img src="https://raw.githubusercontent.com/yoyotv/Image-derain-via-CGAN/master/figures/3_ground_truth.JPG" >|



## Cases that we tried
| Model number | Discriminator Optimizer / Learning rate | Generator Optimizer / Learning rate / beta1 | GAN loss coefficient | VGG loss coefficient | Raw loss coefficient | label switch frequency |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|1| SGD / 0.002 | Adam / 0.002 / 0.9 | 1 | 0.01 | 1 | 10 |
|2| SGD / 0.002 | Adam / 0.0002 / 0.9 | 1 | 0.01 | 1 | 10 |
|3| SGD / 0.002 | Adam / 0.002 / 0.5 | 1 | 0.015 | 10 | 10 |
|4| SGD / 0.002 | Adam / 0.002 / 0.5 | 1 | 0.01 | 1 | 10 |
|5| SGD / 0.002 | Adam / 0.002 / 0.9 | 1 | 0.01 | 1 | NO |
|6| Adam / 0.002 | Adam / 0.002 / 0.9 | 1 | 0.01 | 1 | NO |
|7| SGD / 0.0002 | Adam / 0.0002 / 0.9 | 3 | 0.015 | 15 | 20 |
|8| SGD / 0.002 | Adam / 0.00002 / 0.9 | 3 | 0.015 | 15 | 20 |
|9| SGD / 0.002 | Adam / 0.00001 / 0.9 | 3 | 0.015 | 15 | 20 |
|10| SGD / 0.002 | Adam / 0.0002 / 0.9 | 3 | 0.015 | 15 | 20 |
|12| SGD / 0.002 | Adam / 0.0002 / 0.9 | 1 | 0.01 | 15 | 5 |
|12| SGD / 0.002 | Adam / 0.0002 / 0.5 | 1 | 0.01 | 15 | 5 |
|13| SGD / 0.002 | Adam / 0.0002 / 0.5 | 0.5 | 0.012 | 15 | 5 |
|14| SGD / 0.002 | Adam / 0.0002 / 0.5 | 0.4 | 0.015 | 15 | 5 |


## References

[[1]](https://arxiv.org/pdf/1701.05957v3.pdf) He Zhang, Vishwanath Sindagi, Vishal M. Patel "Image De-raining Using a Conditional Generative Adversarial Network," arXiv:1701.05957v3 

[[2]](https://arxiv.org/ftp/arxiv/papers/1810/1810.09479.pdf) Bharath Raj N., Venkateswaran N, "Single Image Haze Removal using a Generative Adversarial Network," arXiv:1810.09479
 
