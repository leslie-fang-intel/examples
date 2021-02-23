import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import intel_pytorch_extension as ipex
#import _torch_ipex as core

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--ipex', action='store_true', default=False,
                    help='use intel pytorch extension')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disable CUDA')
parser.add_argument('--dnnl', action='store_true', default=False,
                    help='enable Intel_PyTorch_Extension auto dnnl path')
parser.add_argument('--int8', action='store_true', default=False,
                    help='enable ipex int8 path')
parser.add_argument('--jit', action='store_true', default=False,
                    help='enable ipex jit fusionpath')
parser.add_argument('--calibration', action='store_true', default=False,
                    help='doing calibration step')
parser.add_argument('--configure-dir', default='configure.json', type=str, metavar='PATH',
                    help = 'path to int8 configures, default file name is configure.json')
parser.add_argument("--dummy", action='store_true',
                    help="using  dummu data to test the performance of inference")
parser.add_argument('-w', '--warmup-iterations', default=30, type=int, metavar='N',
                    help='number of warmup iterati ons to run')
parser.add_argument('--autocast', action='store_true', default=False,
                    help='enable autocast CPU in torch')

best_acc1 = 0


def main():
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(args)
    if args.ipex:
        import intel_pytorch_extension as ipex

    if args.dnnl:
        ipex.core.enable_auto_dnnl()
    elif args.ipex:
        import intel_pytorch_extension as ipex
        ipex.core.disable_auto_dnnl()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None and args.cuda:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count() if args.cuda else 0
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu
        
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None and args.cuda:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            if args.cuda:
                model.cuda()
                print("create DistributedDataParallel in GPU")
            else:
                print("create DistributedDataParallel in CPU")
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None and args.cuda:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            if args.cuda:
                model.cuda()
        else:
            model = torch.nn.DataParallel(model)
            if args.cuda():
                model.cuda()

    if args.ipex:
        model = model.to(device = ipex.DEVICE)
    # define loss function (criterion) and optimizer

    criterion = nn.CrossEntropyLoss()
    if args.cuda:
        criterion = criterion.cuda(args.gpu)
    elif args.ipex:
        criterion = criterion.to(device = ipex.DEVICE)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None and args.cuda:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None and args.cuda:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.cuda:
        cudnn.benchmark = True

    if args.dummy:
        assert args.evaluate, "please using real dataset if you want run training path"
    if not args.dummy:
        # Data loading code
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None
        #train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    else:
        train_loader = None
        val_loader = None


    if args.evaluate:
        if args.ipex:
            print("using ipex model to do inference\n")
        if args.jit:
            print("running jit fusion path\n")
            script_model = torch.jit.script(model)

        if args.jit:
            validate(val_loader, script_model, criterion, args)
        else:
            validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()

    #conf = ipex.AmpConf(torch.bfloat16) if args.ipex else None
    if args.ipex:
        ipex.enable_auto_mixed_precision(mixed_dtype = torch.bfloat16)

    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        if args.autocast:
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, device=torch.device('mkldnn')):
                output = model(images)
                loss = criterion(output, target)
            output = output.to_dense().to(torch.float32)
        elif args.ipex:
            images = images.to(device = ipex.DEVICE)
        #    with ipex.AutoMixPrecision(conf):
        #        images = images.to(device = ipex.DEVICE)
        #        output = model(images)
        #        loss = criterion(output, target)
        #else:
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    if args.dummy:
        number_iter = 300
    else:
        number_iter = len(val_loader)
    if args.calibration:
        number_iter = 100

    progress = ProgressMeter(
        number_iter,
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    if args.ipex and args.int8 and args.calibration:
        print("runing int8 calibration step\n")
        conf = ipex.AmpConf(torch.int8)
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(val_loader):
                with ipex.AutoMixPrecision(conf, running_mode="calibration"):
                    images = images.to(device = ipex.DEVICE)
                    # compute output
                    output = model(images)
                    loss = criterion(output, target)

                    # measure accuracy and record loss
                    acc1, acc5 = accuracy(output, target, topk=(1, 5))
                    losses.update(loss.item(), images.size(0))
                    top1.update(acc1[0], images.size(0))
                    top5.update(acc5[0], images.size(0))

                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()

                    if i % args.print_freq == 0:
                        progress.display(i)
                    if i == 10:
                        break

            conf.save(args.configure_dir)
            # TODO: this should also be done with the ProgressMeter
            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))
    else:
        if args.ipex:
            if args.int8:
                conf = ipex.AmpConf(torch.int8, args.configure_dir)
                print("running int8 evalation step\n")
            else:
                #conf = ipex.AmpConf(None)
                conf = ipex.AmpConf(torch.bfloat16)
                print("running fp32 evalation step\n")
        if args.dummy:
            if args.ipex:
                with torch.no_grad():
                    print("LeslieDebug: IPEX throughput")
                    for i in range(number_iter):
                        images = torch.randn(args.batch_size, 3, 224, 224).to(device = ipex.DEVICE)
                        target = torch.arange(1, args.batch_size + 1).long().to(device = ipex.DEVICE)
                        with ipex.AutoMixPrecision(conf, running_mode="inference"):
                            if i >= args.warmup_iterations:
                                end = time.time()
                            # compute output
                            #if i == 11:
                            #    with torch.autograd.profiler.profile(use_cuda=False, record_shapes=True) as prof:
                            #        output = model(images)
                            #    print(prof.key_averages().table(sort_by="self_cpu_time_total"))
                            #else:
                            #    output = model(images)

                            output = model(images)
                            
                            if i >= args.warmup_iterations:
                                batch_time.update(time.time() - end)

                            #print(output)
                            loss = criterion(output, target)

                            print("LeslieDebug: Finish one step")

                            # measure accuracy and record loss
                            acc1, acc5 = accuracy(output, target, topk=(1, 5))
                            losses.update(loss.item(), images.size(0))
                            top1.update(acc1[0], images.size(0))
                            top5.update(acc5[0], images.size(0))


                            if i % args.print_freq == 0:
                                progress.display(i)
            else:
                with torch.no_grad():
                    print("LeslieDebug: TORCH throughput")

                    #from torch.utils import mkldnn as mkldnn_utils
                    #model = mkldnn_utils.to_mkldnn(model)
                    #print(model)
                    
                    for i in range(number_iter):
                        images = torch.randn(args.batch_size, 3, 224, 224)
                        target = torch.arange(1, args.batch_size + 1).long()
                        if args.gpu is not None and args.cuda:
                            images = images.cuda(args.gpu, non_blocking=True)
                        if args.cuda:
                            target = target.cuda(args.gpu, non_blocking=True)

                        #images = images.to(torch.bfloat16).to_mkldnn()

                        if i >= args.warmup_iterations:
                            end = time.time()
                        # compute output
                        if args.autocast:
                            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, device=torch.device('mkldnn')):
                                #if i == 11:
                                #    with torch.autograd.profiler.profile(use_cuda=False, record_shapes=True) as prof:
                                #        output = model(images)
                                #    print(prof.key_averages().table(sort_by="self_cpu_time_total"))
                                #    prof.export_chrome_trace("torch_throughput.json")
                                #else:
                                #   output = model(images) 
                                output = model(images)
                                #output = output.to()
                                if i >= args.warmup_iterations:
                                  batch_time.update(time.time() - end)
                                loss = criterion(output, target)
                            output = output.to_dense().to(torch.float)
                        else:
                            output = model(images)
                            if i >= args.warmup_iterations:
                                batch_time.update(time.time() - end)

                            #output = output.to_dense()
                            loss = criterion(output, target)

                        # measure accuracy and record loss
                        #if(output.is_mkldnn())
                        #output = output.to_dense().to(torch.float)
                        
                        print("LeslieDebug: Finish one step")
                        acc1, acc5 = accuracy(output, target, topk=(1, 5))
                        losses.update(loss.item(), images.size(0))
                        top1.update(acc1[0], images.size(0))
                        top5.update(acc5[0], images.size(0))

                        if i % args.print_freq == 0:
                            progress.display(i)
        else:
            if args.ipex:
                with torch.no_grad():
                    print("LeslieDebug: IPEX accuracy")
                    for i, (images, target) in enumerate(val_loader):
                        with ipex.AutoMixPrecision(conf, running_mode="inference"):
                            images = images.to(device = ipex.DEVICE)
                            target = target.to(device = ipex.DEVICE)
                            # compute output
                            if i >= args.warmup_iterations:
                                end = time.time()
                            # compute output
                            output = model(images)
                            if i >= args.warmup_iterations:
                                batch_time.update(time.time() - end)

                            #print(output)
                            loss = criterion(output, target)

                            # measure accuracy and record loss
                            acc1, acc5 = accuracy(output, target, topk=(1, 5))
                            losses.update(loss.item(), images.size(0))
                            top1.update(acc1[0], images.size(0))
                            top5.update(acc5[0], images.size(0))

                            if i % args.print_freq == 0:
                                progress.display(i)
            else:
                with torch.no_grad():
                    end = time.time()
                    print("LeslieDebug: TORCH accuracy")
                    for i, (images, target) in enumerate(val_loader):
                        if args.gpu is not None and args.cuda:
                            images = images.cuda(args.gpu, non_blocking=True)
                        if args.cuda:
                            target = target.cuda(args.gpu, non_blocking=True)
                        # compute output
                        if i >= args.warmup_iterations:
                            end = time.time()
                        # compute output
                        #output = model(images)
                        #print(output)
                        #if i >= args.warmup_iterations:
                        #    batch_time.update(time.time() - end)
                        #loss = criterion(output, target)

                        if args.autocast:
                            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, device=torch.device('mkldnn')):
                                output = model(images)
                                #print(output)
                                if i >= args.warmup_iterations:
                                    batch_time.update(time.time() - end)
                                loss = criterion(output, target)
                            output = output.to_dense().to(torch.float)
                        else:
                            output = model(images)
                            #print(output)
                            if i >= args.warmup_iterations:
                                batch_time.update(time.time() - end)
                            loss = criterion(output, target)
                            #output = output.to(torch.float32)
                        
                        # measure accuracy and record loss
                        
                        
                        acc1, acc5 = accuracy(output, target, topk=(1, 5))
                        losses.update(loss.item(), images.size(0))
                        top1.update(acc1[0], images.size(0))
                        top5.update(acc5[0], images.size(0))

                        if i % args.print_freq == 0:
                            progress.display(i)

        batch_size = args.batch_size
        latency = batch_time.avg / batch_size * 1000
        perf = batch_size / batch_time.avg
        print('inference latency %.3f ms'%latency)
        print('inference performance %.3f fps'%perf)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
