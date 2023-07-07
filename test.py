import os
import time
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from argparse import ArgumentParser
# user
from builders.model_builder import build_model
from builders.dataset_builder import build_dataset_test
from utils.utils import save_predict
from utils.metric import get_iou
from utils.convert_state import convert_state_dict


def test(args, test_loader, model):
    """
    args:
      test_loader: loaded for test dataset
      model: model
    return: class IoU and mean IoU
    """
    # evaluation or test mode
    # 将模型切换到评估模式，，因为在测试过程中，我们不需要进行梯度计算和反向传播。在评估模式下，模型的权重不会被更新。
    # total_batches是test_loader中的批次总数。
    model.eval()
    total_batches = len(test_loader)

    # 创建一个空列表，用于存储模型对测试数据集进行预测的结果。
    data_list = []
    # 使用enumerate()函数遍历test_loader中的数据，返回一个元组 (i, (input, label, size, name))，其中 i 是当前迭代的索引，
    # (input, label, size, name) 是一个元组，包含了当前迭代的输入数据、标签、大小和名称等信息。
    for i, (input, label, size, name) in enumerate(test_loader):
        with torch.no_grad():
            input_var = Variable(input).cuda()
        start_time = time.time()
        output = model(input_var)
        torch.cuda.synchronize()
        time_taken = time.time() - start_time
        print('[%d/%d]  time: %.2f' % (i + 1, total_batches, time_taken))
        output = output.cpu().data[0].numpy()
        gt = np.asarray(label[0].numpy(), dtype=np.uint8)
        output = output.transpose(1, 2, 0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
        data_list.append([gt.flatten(), output.flatten()])
        
        # 如果指定了保存路径args.save，那么将预测的结果保存到指定的路径中。
        # 调用了一个名为save_predict的函数，传入了预测结果output，真实标签gt，图片名称name[0]，
        # 数据集名称args.dataset，保存路径args.save_seg_dir等参数。
        # save_predict函数会将预测结果和真实标签绘制在一起保存为一张图片，并根据参数设置生成彩色或灰度的输出图像。
        # save the predicted image
        if args.save:
            save_predict(output, gt, name[0], args.dataset, args.save_seg_dir,
                         output_grey=False, output_color=True, gt_color=True)

    meanIoU, per_class_iu = get_iou(data_list, args.classes)
    return meanIoU, per_class_iu


def test_model(args):
    """
     main function for testing
     param args: global arguments
     return: None
    """
    print(args)

    if args.cuda:
        print("=====> use gpu id: '{}'".format(args.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        if not torch.cuda.is_available():
            raise Exception("no GPU found or wrong gpu id, please run without --cuda")

    # build the model
    model = build_model(args.model, num_classes=args.classes)

    if args.cuda:
        model = model.cuda()  # using GPU for inference
        cudnn.benchmark = True

    if args.save:
        if not os.path.exists(args.save_seg_dir):
            os.makedirs(args.save_seg_dir)

    # load the test set
    datas, testLoader = build_dataset_test(args.dataset, args.num_workers)

    if not args.best:
        if args.checkpoint:
            if os.path.isfile(args.checkpoint):
                print("=====> loading checkpoint '{}'".format(args.checkpoint))
                checkpoint = torch.load(args.checkpoint)
                model.load_state_dict(checkpoint['model'])
                # model.load_state_dict(convert_state_dict(checkpoint['model']))
            else:
                print("=====> no checkpoint found at '{}'".format(args.checkpoint))
                raise FileNotFoundError("no checkpoint found at '{}'".format(args.checkpoint))

        print("=====> beginning validation")
        print("validation set length: ", len(testLoader))
        mIOU_val, per_class_iu = test(args, testLoader, model)
        print(mIOU_val)
        print(per_class_iu)

    # Get the best test result among the last 10 model records.
    else:
        if args.checkpoint:
            if os.path.isfile(args.checkpoint):
                dirname, basename = os.path.split(args.checkpoint)
                epoch = int(os.path.splitext(basename)[0].split('_')[1])
                mIOU_val = []
                per_class_iu = []
                for i in range(epoch - 9, epoch + 1):
                    basename = 'model_' + str(i) + '.pth'
                    resume = os.path.join(dirname, basename)
                    checkpoint = torch.load(resume)
                    model.load_state_dict(checkpoint['model'])
                    print("=====> beginning test the" + basename)
                    print("validation set length: ", len(testLoader))
                    mIOU_val_0, per_class_iu_0 = test(args, testLoader, model)
                    mIOU_val.append(mIOU_val_0)
                    per_class_iu.append(per_class_iu_0)

                index = list(range(epoch - 9, epoch + 1))[np.argmax(mIOU_val)]
                print("The best mIoU among the last 10 models is", index)
                print(mIOU_val)
                per_class_iu = per_class_iu[np.argmax(mIOU_val)]
                mIOU_val = np.max(mIOU_val)
                print(mIOU_val)
                print(per_class_iu)

            else:
                print("=====> no checkpoint found at '{}'".format(args.checkpoint))
                raise FileNotFoundError("no checkpoint found at '{}'".format(args.checkpoint))

    # Save the result
    if not args.best:
        model_path = os.path.splitext(os.path.basename(args.checkpoint))
        args.logFile = 'test_' + model_path[0] + '.txt'
        logFileLoc = os.path.join(os.path.dirname(args.checkpoint), args.logFile)
    else:
        args.logFile = 'test_' + 'best' + str(index) + '.txt'
        logFileLoc = os.path.join(os.path.dirname(args.checkpoint), args.logFile)

    # Save the result
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Mean IoU: %.4f" % mIOU_val)
        logger.write("\nPer class IoU: ")
        for i in range(len(per_class_iu)):
            logger.write("%.4f\t" % per_class_iu[i])
    logger.flush()
    logger.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', default="ELANet", help="model name: Context Guided Network (CGNet)")
    parser.add_argument('--dataset', default="camvid", help="dataset: cityscapes or camvid")
    parser.add_argument('--num_workers', type=int, default=4, help="the number of parallel threads")
    parser.add_argument('--batch_size', type=int, default=1,
                        help=" the batch_size is set to 1 when evaluating or testing")
    parser.add_argument('--checkpoint', type=str,
                        default="D:\ELANet-master\checkpoint_Shuffle\camvid\ELANetbs4gpu1_trainval\model_1000.pth",
                        help="use the file to load the checkpoint for evaluating or testing ")
    parser.add_argument('--save_seg_dir', type=str, default="./result/",
                        help="saving path of prediction result")
    parser.add_argument('--best', action='store_true', help="Get the best result among last few checkpoints")
    parser.add_argument('--save', action='store_true', help="Save the predicted image")
    parser.add_argument('--cuda', default=True, help="run on CPU or GPU")
    parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
    args = parser.parse_args()

    args.save_seg_dir = os.path.join(args.save_seg_dir, args.dataset, args.model)

    if args.dataset == 'cityscapes':
        args.classes = 19
    elif args.dataset == 'camvid':
        args.classes = 11
    else:
        raise NotImplementedError(
            "This repository now supports two datasets: cityscapes and camvid, %s is not included" % args.dataset)

    test_model(args)
