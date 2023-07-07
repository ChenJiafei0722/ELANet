# ELANet: An efficiently lightweight asymmetrical network for real-time semantic segmentation


### Dataset
You need to download the two dataset——CamVid and Cityscapes, and put the files in the `dataset` folder with following structure.
```
├── camvid
|    ├── train
|    ├── test
|    ├── val 
|    ├── trainannot
|    ├── testannot
|    ├── valannot
|    ├── camvid_trainval_list.txt
|    ├── camvid_train_list.txt
|    ├── camvid_test_list.txt
|    └── camvid_val_list.txt
├── cityscapes
|    ├── gtCoarse
|    ├── gtFine
|    ├── leftImg8bit
|    ├── cityscapes_trainval_list.txt
|    ├── cityscapes_train_list.txt
|    ├── cityscapes_test_list.txt
|    └── cityscapes_val_list.txt           
```

### Training

- You can run: `python train.py -h` to check the detail of optional arguments.
Basically, in the `train.py`, you can set the dataset, train type, epochs and batch size, etc.
```
python train.py --dataset ${camvid, cityscapes} --train_type ${train, trainval} --max_epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --lr ${LR} --resume ${CHECKPOINT_FILE}
```
- training on Cityscapes train set
```
python train.py --dataset cityscapes
```
- training on CamVid train and val set
```
python train.py --dataset camvid --train_type trainval --max_epochs 1000 --lr 1e-3 --batch_size 16
```
- During training course, every 50 epochs, we will record the mean IoU of train set, validation set and training loss to draw a plot, so you can check whether the training process is normal.

### Testing
- After training, the checkpoint will be saved at `checkpoint` folder, you can use `test.py` to get the result.
```
python test.py --dataset ${camvid, cityscapes} --checkpoint ${CHECKPOINT_FILE}
```
### Evaluation
- For those dataset that do not provide label on the test set (e.g. Cityscapes), you can use `predict.py` to save all the output images, then submit to official webpage for evaluation.
```
python predict.py --checkpoint ${CHECKPOINT_FILE}
```


### Inference Speed
- You can run the `eval_fps.py` to test the model inference speed, input the image size such as `512,1024`.
```
python eval_fps.py 512,1024
```


