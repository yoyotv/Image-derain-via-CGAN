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

2. Replace the image.c under darknet/src, the new image.c will help us generate a txt file containing the predict result. We need this txt file when we are calculating the total price. 

3. Put the cfg/supermarket folder files  under darknet/cfg

4. Put the [Pretrain weights](https://drive.google.com/open?id=1vOuW_z4SHUWJDOcIcXVpYHogKqh7j7ou) under darknet/backup/supermarket/

### Pre-train model
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
|11| SGD / 0.002 | Adam / 0.0002 / 0.9 | 1 | 0.01 | 15 | 20 |
|12| SGD / 0.002 | Adam / 0.0002 / 0.9 | 1 | 0.01 | 15 | 5 |
|13| SGD / 0.002 | Adam / 0.0002 / 0.5 | 0.5 | 0.015 | 15 | 5 |




## Activate the system

Right now we are under supermarknet folder

```
sh run.sh
```
1. The system will read the image in the test_image, you may change the ipout image in "run.sh" line 4.

2. Darknet engine will apply YOLOv3 to predict the locationa and the class of the input image, then save the result in "current record" folder which named "predictions.txt"

3. System will use "calculate.py" to read the predictions and calculate the total price of image. (You may revise the price list in "calculate.py")

4. The final price will be printed on the command window and saved with the prediction in a txt file and saved under records with a format Year_Month_Date_Hour:Minute:Second.txt.

5. The result shouls loo like this

   <img src="https://raw.githubusercontent.com/yoyotv/YOLO-project/master/supermarket_checkout_system/demo.png" >

## Training 

Go to darknet folder

1. Use `pycocoDemo.py` to convert original coco format label into darknet format label 

    1.1. Go to https://github.com/cocodataset/cocoapi.git and install the coco api
  
    1.2. Put the training image under coco/images
  
    1.3. Put `pycocoDemo.py` under PythonAPI and run `python pycocoDemo.py`
  
    1.4. The darknet format label will be generated under coco/images

2. Create an folder named D2S under /darknet/data/

3. Copy all files (images and label.txt) to coco/images/D2S

4. Copy all darknet format labels to a new folder named "tem" under coco

5. Use "gen_list" to generate training list, run ``` python gen_list.py```.

6. There will be a train_file_list.txt under darknet/cfg/supermarket/.

7. Run

```
./darknet detector train /home/dl-linux/darknet/cfg/supermarket/obj.data /home/dl-linux/darknet/cfg/supermarket/yolov3-tiny.cfg /home/dl-linux/darknet/cfg/supermarket/yolov3-tiny.conv.15
```

8. The weights after training will place under darknet/backup.


