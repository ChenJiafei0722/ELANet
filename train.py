import os
import time
import torch
import torch.nn as nn
import timeit
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from argparse import ArgumentParser
# argparse 是 Python 自带的一个命令行解析库，它可以方便地对命令行输入的参数进行解析和处理。
# 在这段代码中，通过 from argparse import ArgumentParser 导入了 ArgumentParser 类，用于解析命令行参数。
# user
from builders.model_builder import build_model
from builders.dataset_builder import build_dataset_train
from utils.utils import setup_seed, init_weight, netParams
from utils.metric import get_iou
from utils.loss import CrossEntropyLoss2d, ProbOhemCrossEntropy2d
from utils.lr_scheduler import WarmupPolyLR
from utils.convert_state import convert_state_dict

GLOBAL_SEED = 1234


def val(args, val_loader, model):
    """
    args:
      val_loader: loaded for validation dataset
      model: model
    return: mean IoU and IoU class
    """
    # args:参数、val_loader代表验证数据集的加载器，model代表待评估的模型。
    # evaluation mode
    # 这两行代码的作用是将模型设置为evaluation模式，同时获取validation数据集的总批次数。
    model.eval()
    total_batches = len(val_loader)

    data_list = []
    # 在函数内部，首先将模型置于评估模式
    # 然后使用enumerate迭代遍历验证数据集中的每个批次，对于每个批次的输入数据input，将其传入模型中，并记录评估时间。
    # 将模型输出转移到CPU上，并将输出变量output的值转换为NumPy数组。
    # 接下来，将output数组的形状转换为(H, W)，并使用np.argmax()函数计算出每个像素点的预测类别。
    # 最后，将预测值和真实值展平转换为一维数组并存储在列表data_list中。最终 data_list 中保存了所有样本的预测结果和标签。
    # 在遍历完整个验证数据集之后，调用get_iou()函数计算所有批次的平均交并比(meanIoU)和每个类别的交并比(per_class_iu)，并将它们返回。

    # 使用python的迭代器机制，每次从 val_loader 中获取一个 batch 的数据。
    # 其中 input 是输入数据的张量，label 是对应的标签，size 是输入数据的尺寸，name 是数据的名称（如果有的话）。
    # 这些数据都是从 val_loader 中读取的。在 enumerate 函数的帮助下，可以得到当前 batch 的序号 i 和 total_batches，即总共有多少个 batch。
    # 首先，使用 time 模块记录当前时间，然后使用模型预测输入数据的输出。最后，计算模型的推断时间，即当前时间减去之前记录的时间。
    # 将模型输出和真实标签转换为Numpy数组。输出变量是包含输入图像的预测分割图的张量。标签变量是包含输入图像的真实分割图的张量。
    # cpu（）方法将输出张量从GPU内存移动到CPU内存，数据属性提取张量的数据，numpy（）方法将张量转换为NumPy数组。
    # gt变量也使用numpy（）方法转换为NumPy数组。
    # np.asarray方法的dtype参数指定结果NumPy数组的数据类型，设置为np.uint8。
    for i, (input, label, size, name) in enumerate(val_loader):
        with torch.no_grad():
            input_var = Variable(input).cuda()
        start_time = time.time()
        output = model(input_var)
        time_taken = time.time() - start_time
        print("[%d/%d]  time: %.2f" % (i + 1, total_batches, time_taken))
        output = output.cpu().data[0].numpy()
        gt = np.asarray(label[0].numpy(), dtype=np.uint8)
        output = output.transpose(1, 2, 0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
        data_list.append([gt.flatten(), output.flatten()])

    meanIoU, per_class_iu = get_iou(data_list, args.classes)
    return meanIoU, per_class_iu


def train(args, train_loader, model, criterion, optimizer, epoch):


    """
    args:
       train_loader: loaded for training dataset  训练数据集的加载
       model: model
       criterion: loss function   损失函数
       optimizer: optimization algorithm, such as ADAM or SGD   优化器
       epoch: epoch number  epoch数
    return: average loss, per class IoU, and mean IoU
    """
    # 模型倍设置为训练模式
    # 一个空的列表epoch_loss被初始化，用于保存每个batch的损失。总共的batch数被记录下来并打印出来。
    model.train()
    epoch_loss = []
    # 计算训练数据集及共有多少个batch。train_loader是一个迭代器，每个迭代器产生一个batch的数据用于训练模型
    # 使用len()函数获取训练数据集的批次数total_batches，len(train_loader) 就是迭代器中batch的数量。
    # 定义变量st并使用time.time()获取当前时间作为起始时间。
    total_batches = len(train_loader)
    print("=====> the number of iterations per epoch: ", total_batches)
    st = time.time()

    for iteration, batch in enumerate(train_loader, 0):
        # 在训练数据集加载器中，使用enumerate()函数循环遍历每个批次batch。
        # 对于每个批次，设置了几个变量，包括每个epoch中的迭代总次数iteration，当前迭代的次数cur_iter
        # 迭代的总次数max_iter，当前迭代的学习率lr，并创建一个学习率调度程序scheduler。
        # scheduler是调整学习率的工具，根据WarmupPolyLR算法进行调整。lr表示当前迭代的学习率。
        args.per_iter = total_batches
        args.max_iter = args.max_epochs * args.per_iter
        args.cur_iter = epoch * args.per_iter + iteration
        # 在训练过程中，每个epoch开始时，根据当前迭代次数、总迭代次数和学习率等参数，计算并更新优化器的学习率。
        # 在这里，使用了WarmupPolyLR函数实现学习率调度器，并将当前学习率赋值给lr变量。
        # warmup_factor=1.0 / 3 表示学习率的预热系数，即在前一段时间内使用较小的学习率以确保网络在训练初期不会过度拟合，这里的预热系数为初始学习率的1/3。
        # warmup_iters=500表示前500个迭代次数内学习率线性增加
        # power=0.9指的是PolyLR的参数，表示学习率的下降速率。
        scheduler = WarmupPolyLR(optimizer, T_max=args.max_iter, cur_iter=args.cur_iter, warmup_factor=1.0 / 3,
                                 warmup_iters=500, power=0.9)
        lr = optimizer.param_groups[0]['lr']

        # 将一个batch的数据从数据加载器中获取并转换成可训练的张量，并将其传递给模型以获得输出。具体来说：
        # images, labels, _, _ = batch：将batch中的四个元素解包并赋值给images和labels变量，而忽略其他两个元素。
        # 通常，这些元素分别是输入图像张量、标签张量、输入图像的路径和标签的路径。
        # images = Variable(images).cuda()：将输入图像张量转换为PyTorch可优化的变量类型并将其移动到GPU上进行加速计算。
        # labels = Variable(labels.long()).cuda()：将标签张量转换为PyTorch可优化的变量类型并将其移动到GPU上进行加速计算。
        # output = model(images)：将输入图像张量传递给模型计算以获得输出。
        # 将模型的输出 output 与标签 labels 传递给损失函数 criterion，计算出当前的损失值 loss。
        # 调用 scheduler.step() 函数，更新学习率。这里使用的是 WarmupPolyLR 调度器，其会根据当前迭代次数和总迭代次数，以及其他一些参数，动态调整学习率。
        # 将梯度置零，然后调用 loss.backward() 函数，计算出每个模型参数的梯度。
        # 调用 optimizer.step() 函数，更新模型参数，使其朝着减小损失值的方向移动。
        # 将当前的损失值 loss 添加到 epoch_loss 列表中，以便在训练完成后计算平均损失值。
        # 计算当前迭代的时间 time_taken，以便在输出中打印出来。现在的时间 - 迭代开始的时间

        start_time = time.time()
        images, labels, _, _ = batch
        images = Variable(images).cuda()
        labels = Variable(labels.long()).cuda()
        output = model(images)
        loss = criterion(output, labels)
        scheduler.step()
        optimizer.zero_grad()  # set the grad to zero
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())
        time_taken = time.time() - start_time

        print('=====> epoch[%d/%d] iter: (%d/%d) \tcur_lr: %.6f loss: %.3f time:%.2f' % (epoch + 1, args.max_epochs,
                                                                                         iteration + 1, total_batches,
                                                                                         lr, loss.item(), time_taken))
    # time_taken_epoch表示已经训练的所有epoch所花费的时间（即从第一个epoch到当前epoch所花费的总时间）
    # remain_time表示还剩下多少时间来完成整个训练（即从当前epoch到最后一个epoch所需要的时间）
    # 其中，remain_time的计算方法是将已经训练的时间乘以剩余的epoch数（args.max_epochs - 1 - epoch）。
    time_taken_epoch = time.time() - st
    remain_time = time_taken_epoch * (args.max_epochs - 1 - epoch)
    # 将剩余训练时间转换为时分秒的格式，并输出到控制台上。
    m, s = divmod(remain_time, 60)
    h, m = divmod(m, 60)
    print("Remaining training time = %d hour %d minutes %d seconds" % (h, m, s))
    # 这段代码计算了本轮 epoch 训练的平均损失，将所有的 batch 损失求和，然后除以 batch 数量得到平均损失。
    # len(epoch_loss) 表示 epoch_loss 列表的长度，即列表中元素的个数。
    # 在这个代码段中，epoch_loss 存储了当前 epoch 中每个 batch 的损失值，通过取平均值来计算当前 epoch 的平均损失值。
    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)

    return average_epoch_loss_train, lr


def train_model(args):
    # 下面代码意思是在控制台中打印出参数 args 的值，它是一个字典类型的对象，包含了整个模型训练的超参数设置和数据集路径等信息。
    """
    args:
       args: global arguments
    """
    # 函数内部的h, w = map(int, args.input_size.split(','))将参数args中的input_size用逗号分隔开
    # 分别赋值给h和w，并将它们转换为整数类型。接着打印出输入的大小。最后，打印出参数信息。
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    print("=====> input size:{}".format(input_size))

    print(args)
    # 判断是否使用GPU训练
    if args.cuda:
        print("=====> use gpu id: '{}'".format(args.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    # set the seed
    # 设置全局随机种子，确保每次训练的结果是一致的，即在运行模型时使用相同的随机数生成器初始状态，确保模型在不同运行环境下具有相同的结果
    setup_seed(GLOBAL_SEED)
    print("=====> set Global Seed: ", GLOBAL_SEED)

    # 这段代码是在启用CUDA加速并打印构建神经网络的信息。
    # cudnn.enabled = True是启用cudnn加速，它可以提高神经网络的训练速度。
    # print("=====> building network")则是打印构建神经网络的信息，以便在代码运行时进行调试和检查。
    cudnn.enabled = True
    print("=====> building network")

    # build the model and initialization
    # 根据指定的参数args.model和args.classes创建一个ELANet神经网络模型。
    # init_weight是一个自定义的函数，用于初始化模型权重。
    # nn.init.kaiming_normal_是一个PyTorch内置的函数，用于权重初始化的方法，用于以Kaiming正态分布方式初始化权重。
    # nn.BatchNorm2d是另一个PyTorch内置的函数，表示要使用的归一化层类型，用于实现批量归一化（batch normalization）操作。
    # 1e-3: 归一化层中 epsilon 参数的值
    # 0.1: 归一化层中 momentum 参数的值
    # mode='fan_in'表示计算权重时使用Kaiming正态分布中的“fan_in”模式。
    model = build_model(args.model, num_classes=args.classes)
    init_weight(model, nn.init.kaiming_normal_,
                nn.BatchNorm2d, 1e-3, 0.1,
                mode='fan_in')

    # 输出神经网络的总参数数量
    # %.2f M是以百万为单位展示的参数数量。
    # 例如，如果total_parameters为2000000，则输出“the number of parameters: 2000000 ==> 2.00 M”。
    print("=====> computing network parameters and FLOPs")
    total_paramters = netParams(model)
    print("the number of parameters: %d ==> %.2f M" % (total_paramters, (total_paramters / 1e6)))

    # load data and data augmentation
    # 使用 build_dataset_train() 函数创建了训练集和验证集的数据集，并将结果分别赋值给了 trainLoader 和 valLoader 变量。
    # build_dataset_train() 函数的输入参数包括数据集名称、输入图像尺寸、批量大小、训练类型、随机缩放、随机镜像和工作进程数量等
    datas, trainLoader, valLoader = build_dataset_train(args.dataset, input_size, args.batch_size, args.train_type,
                                                        args.random_scale, args.random_mirror, args.num_workers)

    # 打印数据集的统计信息，包括数据集中每个类别的权重（classWeights），以及数据集的均值（mean）和标准差（std）
    print('=====> Dataset statistics')
    print("data['classWeights']: ", datas['classWeights'])
    print('mean and std: ', datas['mean'], datas['std'])

    # define loss function, respectively 根据不同任务定义不同的损失函数
    # 将一个 NumPy 数组 datas['classWeights'] 转换为 PyTorch 的 Tensor 类型，并将其赋值给变量 weight。
    # datas['classWeights'] 是数据集中每个类别的权重，该权重在训练分类器时用于计算损失函数的加权值。
    weight = torch.from_numpy(datas['classWeights'])
    # 数据集是CamVid，则使用CrossEntropyLoss2d作为损失函数，传入权重weight和忽略标签ignore_label参数。
    # 数据集是Cityscapes，则使用ProbOhemCrossEntropy2d作为损失函数
    if args.dataset == 'camvid':
        criteria = CrossEntropyLoss2d(weight=weight, ignore_label=ignore_label)
    elif args.dataset == 'cityscapes':

        min_kept = int(args.batch_size // len(args.gpus) * h * w // 16)
        criteria = ProbOhemCrossEntropy2d(use_weight=True, ignore_label=ignore_label,
                                          thresh=0.7, min_kept=min_kept)
    else:
        raise NotImplementedError(
            "This repository now supports two datasets: cityscapes and camvid, %s is not included" % args.dataset)
    # 这段代码的意思是将Pytorch模型和损失函数移动到GPU上训练
    if args.cuda:
        criteria = criteria.cuda()
        if torch.cuda.device_count() > 1:
            print("torch.cuda.device_count()=", torch.cuda.device_count())
            args.gpu_nums = torch.cuda.device_count()
            model = nn.DataParallel(model).cuda()  # multi-card data parallel
        else:
            args.gpu_nums = 1
            print("single GPU for training")
            model = model.cuda()  # 1-card data parallel
    # 根据参数生成保存模型文件的目录路径
    args.savedir = (args.savedir + args.dataset + '/' + args.model + 'bs'
                    + str(args.batch_size) + 'gpu' + str(args.gpu_nums) + "_" + str(args.train_type) + '/')

    # 如果指定的保存模型的路径 args.savedir 不存在，则创建该目录。os.makedirs 函数可以递归创建目录。
    # 同时，将变量 start_epoch 的值初始化为0。
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    start_epoch = 0

    # continue training
    # 如果args.resume为True，则会判断是否存在已保存的模型权重文件，若存在则加载该权重文件，同时将训练起始epoch设置为该文件中保存的epoch数。
    # 加载权重文件的过程中，调用load_state_dict方法将权重文件中保存的模型参数赋值给模型。
    # 如果文件不存在，则输出相应提示信息。
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
            # model.load_state_dict(convert_state_dict(checkpoint['model']))
            print("=====> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=====> no checkpoint found at '{}'".format(args.resume))

    # 可用于优化 CUDA 可用 GPU 上卷积神经网络的性能。
    # 当设置为 True 时，它启用 cuDNN 的自动调谐器，根据当前硬件选择最佳卷积操作算法，这可能会导致显著的性能提升。
    model.train()
    cudnn.benchmark = True

    logFileLoc = args.savedir + args.logFile
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Parameters: %s Seed: %s" % (str(total_paramters), GLOBAL_SEED))
        logger.write("\n%s\t\t%s\t%s\t%s" % ('Epoch', 'Loss(Tr)', 'mIOU (val)', 'lr'))
    logger.flush()

    # define optimization criteria
    # 根据训练所使用的数据集选择不同的优化器，并定义了优化器的一些参数。
    if args.dataset == 'camvid':
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), args.lr, (0.9, 0.999), eps=1e-08, weight_decay=2e-4)

    elif args.dataset == 'cityscapes':
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()), args.lr, momentum=0.9, weight_decay=1e-4)
    # 初始化三个空列表存储每个epoch的训练损失、每个epoch的epoch号码、每个epoch的验证mIOU分数。
    lossTr_list = []
    epoches = []
    mIOU_val_list = []

    print('=====> beginning training')
    # 使用一个for循环遍历多个epochs，从start_epoch变量开始，到args.max_epochs变量结束。
    for epoch in range(start_epoch, args.max_epochs):
        # training
        # 对于每个epoch，代码使用train()函数和参数args, trainLoader, model, criteria, optimizer和epoch进行训练。
        # train()函数在训练数据上训练模型一个epoch，并返回该epoch的训练损失和学习率。
        # 然后，代码使用.append()方法将该epoch的训练损失添加到lossTr_list列表中。
        # 总之，该代码训练模型多个epochs，将每个epoch的训练损失存储在一个列表中，并更新每个epoch的学习率。
        lossTr, lr = train(args, trainLoader, model, criteria, optimizer, epoch)
        lossTr_list.append(lossTr)

        # validation
        # 该代码检查当前epoch是否是50的倍数或者是否为最后一个epoch，通过使用取模运算符并将其与零比较，或者检查它是否是最后一个epoch。
        # 如果满足这个条件，它使用.append()方法将epoch数附加到epoches列表中。
        #
        # 然后，该代码使用参数args、valLoader和model调用val()函数。val()函数对验证集执行验证并返回mIOU分数和每个类别的IoU值。
        # 然后，该代码使用.append()方法将mIOU分数附加到mIOU_val_list列表中。
        #
        # 总之，该代码对模型进行某些epochs的验证，将每个验证的epoch数和mIOU分数存储在列表中。
        if epoch % 50 == 0 or epoch == (args.max_epochs - 1):
            epoches.append(epoch)
            mIOU_val, per_class_iu = val(args, valLoader, model)
            mIOU_val_list.append(mIOU_val)
            # record train information
            # 将训练和验证结果写入日志文件并打印到控制台
            # logger.write() 函数将格式化的字符串写入日志文件。该字符串包括当前轮次的轮次编号、训练损失、验证的 mIOU 分数和学习率。
            #
            # logger.flush() 函数强制将日志文件写入磁盘。
            #
            # print() 函数将一条消息打印到控制台，显示当前轮次的轮次编号、训练损失、验证的 mIOU 分数和学习率。
            #
            # 总之，该代码记录和打印当前轮次的结果，包括轮次编号、训练损失、验证的 mIOU 分数和学习率。
            logger.write("\n%d\t\t%.4f\t\t%.4f\t\t%.7f" % (epoch, lossTr, mIOU_val, lr))
            logger.flush()
            print("Epoch : " + str(epoch) + ' Details')
            print("Epoch No.: %d\tTrain Loss = %.4f\t mIOU(val) = %.4f\t lr= %.6f\n" % (epoch,
                                                                                        lossTr,
                                                                                        mIOU_val, lr))
        else:
            # record train information
            logger.write("\n%d\t\t%.4f\t\t\t\t%.7f" % (epoch, lossTr, lr))
            logger.flush()
            print("Epoch : " + str(epoch) + ' Details')
            print("Epoch No.: %d\tTrain Loss = %.4f\t lr= %.6f\n" % (epoch, lossTr, lr))

        # save the model
        # 将训练好的模型保存到文件中。文件名是基于 epoch 数和 args 变量中指定的保存目录生成的。
        # 将 model_file_name 变量设置为字符串，它由保存目录、字符串 "/model_"、当前 epoch 数量加一和字符串 ".pth" 连接而成，
        # 生成的文件名包含了 epoch 数量和 .pth 文件扩展名。
        model_file_name = args.savedir + '/model_' + str(epoch + 1) + '.pth'
        state = {"epoch": epoch + 1, "model": model.state_dict()}

        # 这段代码将训练好的模型状态保存到一个文件中，文件名基于 epoch 数量和保存目录。
        # 如果当前 epoch 的数量大于等于 args.max_epochs 减去 10 或可以被 20 整除，则保存模型。
        # 如果当前 epoch 数量大于等于 args.max_epochs 减去 10，则保存模型。
        # 这是因为在训练结束前的最后 10 个 epoch 中，保存每个 epoch 的模型状态对于后续的测试和推理非常有用。
        # 如果当前 epoch 的数量可以被 20 整除，则保存模型。
        # 这是为了减少文件数量和占用空间，因为每个 epoch 都保存模型状态可能会产生大量文件。
        if epoch >= args.max_epochs - 10:
            torch.save(state, model_file_name)
        elif not epoch % 20:
            torch.save(state, model_file_name)

        # draw plots for visualization
        if epoch % 50 == 0 or epoch == (args.max_epochs - 1):
            # Plot the figures per 50 epochs
            # 生成了一个平均训练损失与训练轮数的图表，并将其保存为 .png 图像文件
            # plt.subplots() 函数被调用以创建一个图形对象和一个轴对象。
            # figsize 参数指定图形的大小，单位为英寸。

            fig1, ax1 = plt.subplots(figsize=(11, 8))

            # ax1.plot() 函数被调用以绘制训练损失与训练轮数的图表。
            # x 轴是从起始轮数到当前轮数的轮数范围，y 轴是存储在 lossTr_list 中的相应训练损失值。
            # ax1.set_title()、ax1.set_xlabel() 和 ax1.set_ylabel() 函数被用于设置图表的标题、x 轴标签和 y 轴标签。
            ax1.plot(range(start_epoch, epoch + 1), lossTr_list)
            ax1.set_title("Average training loss vs epochs")
            ax1.set_xlabel("Epochs")
            ax1.set_ylabel("Current loss")

            # plt.savefig() 函数被调用以将图表保存为 .png 图像文件，并将其存储在指定的保存目录中。
            plt.savefig(args.savedir + "loss_vs_epochs.png")

            # 最后，plt.clf() 函数被调用以清除图形，以便为下一张图表做准备。
            plt.clf()

            fig2, ax2 = plt.subplots(figsize=(11, 8))

            ax2.plot(epoches, mIOU_val_list, label="Val IoU")
            ax2.set_title("Average IoU vs epochs")
            ax2.set_xlabel("Epochs")
            ax2.set_ylabel("Current IoU")
            # plt.legend()函数用于添加图例，loc='lower right'参数指定图例的位置在图表的右下角。
            plt.legend(loc='lower'
                           ' right')

            plt.savefig(args.savedir + "iou_vs_epochs.png")

            # 用于关闭所有打开的matplotlib图形窗口。调用函数plt.close('all')来关闭所有由matplotlib创建的图形窗口。
            plt.close('all')

    # 关闭了之前使用Python logging库打开的日志文件处理器。
    # 关闭logger是一个良好的习惯，以确保日志消息被正确记录并存储在文件中。
    logger.close()


if __name__ == '__main__':
    # 记录程序开始运行的时间，用于计算训练的时间
    start = timeit.default_timer()
    # 使用 ArgumentParser 对象解析命令行参数。这些参数包括模型名称、数据集名称、训练类型、最大 epoch 数、输入图像尺寸、是否随机镜像和缩放、
    # 批量大小、学习率、保存模型快照的目录等等。
    parser = ArgumentParser()
    parser.add_argument('--model', default="ELANet", help="model name: Context Guided Network (CGNet)")
    parser.add_argument('--dataset', default="cityscapes", help="dataset: cityscapes or camvid")
    parser.add_argument('--train_type', type=str, default="trainval",
                        help="ontrain for training on train set, ontrainval for training on train+val set")
    parser.add_argument('--max_epochs', type=int, default=1500,
                        help="the number of epochs: 300 for train set, 350 for train+val set")
    parser.add_argument('--input_size', type=str, default="512,1024", help="input size of model")
    parser.add_argument('--random_mirror', type=bool, default=True, help="input image random mirror")
    parser.add_argument('--random_scale', type=bool, default=True, help="input image resize 0.5 to 2")
    parser.add_argument('--num_workers', type=int, default=4, help=" the number of parallel threads")
    parser.add_argument('--lr', type=float, default=1e-3, help="initial learning rate4.5e-2")
    parser.add_argument('--batch_size', type=int, default=4, help="the batch size is set to 16 for 2 GPUs")
    parser.add_argument('--savedir', default="./checkpoint_test/", help="directory to save the model snapshot")
    parser.add_argument('--resume', type=str, default="",
                        help="use this file to load last checkpoint for continuing training")
    # 指定用于加载最后一个检查点以继续训练的文件
    parser.add_argument('--classes', type=int, default=19,
                        help="the number of classes in the dataset. 19 and 11 for cityscapes and camvid, respectively")
    parser.add_argument('--logFile', default="log.txt", help="storing the training and validation logs")
    parser.add_argument('--cuda', type=bool, default=True, help="running on CPU or GPU")
    parser.add_argument('--gpus', type=str, default="0", help="default GPU devices (0,1)")
    # 从命令行解析参数，将这些参数保存在args中，以便后续使用
    args = parser.parse_args()
    # 这行代码定义了 ignore_label 变量，它的值被设置为 255。
    # 在语义分割任务中，通常将所有标签值都转换为一个介于 0 到 (类别数-1) 的整数，以方便训练和评估。
    # ignore_label 表示应该忽略的标签值，例如在 Cityscapes 数据集中，像素标签 255 表示未标记区域，因此在训练和评估时应该忽略它们。
    # 设置了在CamVid数据集中要被忽略的像素标签值为11。这个值在训练神经网络时用于指示哪些像素应该被忽略，即不应该被用于计算损失函数的像素。
    # 在CamVid数据集中，标签11表示“天空”，这通常是不需要被分类的背景区域。
    if args.dataset == 'cityscapes':
        args.classes = 19
        args.input_size = '512,1024'
        ignore_label = 255
    elif args.dataset == 'camvid':
        args.classes = 11
        args.input_size = '360,480'
        ignore_label = 11
    else:
        raise NotImplementedError(
            "This repository now supports two datasets: cityscapes and camvid, %s is not included" % args.dataset)
    # 调用train_model函数训练模型，训练结束后，记录结束时间，计算训练时间并打印
    train_model(args)
    end = timeit.default_timer()
    hour = 1.0 * (end - start) / 3600
    minute = (hour - int(hour)) * 60
    print("training time: %d hour %d minutes" % (int(hour), int(minute)))
