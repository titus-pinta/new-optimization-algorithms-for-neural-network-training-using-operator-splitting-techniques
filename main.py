import torch
import torch.nn.functional as F

import argparse
import optim

import datetime
import dill
import sys
import os
import time

import models
import dataloader
from test import *
from train import *

import view


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
                        help='learning rate (default: 0.00001)')
    parser.add_argument('--momentum', type=float, default=0.0, metavar='M',
                        help='SGD momentum (default: 0.0)')
    parser.add_argument('--k', type=float, default=2, metavar="K",
                        help='k parameter for SSA algorithm (default: 2)')
    parser.add_argument('--rho', type=float, default=0.9, metavar='R',
                        help='alpha parameter for the Ada based running average (default: 0.9)')
    parser.add_argument('--eps', type=float, default=1e-8, metavar='E',
                        help='eps parameter for the Ada division by 0 correction (default: 1e-8)')
    parser.add_argument('--optim', default='SGD',
                        help='Optimiser to use (default: SGD)', metavar='O')
    parser.add_argument('--loss', default=None, metavar='L',
                        help='Loss function (default: nll for MNIST, cross-entropy for cifar10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--stoch', action='store_true', default=False,
                        help='use stochastic gradient computation')
    parser.add_argument('--cifar10', action='store_true',
                        default=False, help='Use Cifar10 not MNIST')
    parser.add_argument('--fash', action='store_true',
                        default=False, help='Use MNIST fashion not MNIST')
    parser.add_argument('--imdb', action='store_true',
                        default=False, help='Use IMDB bot MNIST')
    parser.add_argument('--optimhelp', default=False,
                        action='store_true', help='Print optim options')
    parser.add_argument('--losshelp', default=False,
                        action='store_true', help='Print loss options')

    args = parser.parse_args()

    if args.optimhelp:
        optims = dir(optim)
        for option in optims:
            if '_' not in option:
                print(option)
        exit()
    if args.losshelp:
        losses = dir(torch.nn)
        for option in losses:
            if 'Loss' in option:
                print(option)
        exit()

    print('Will train for {} epochs with a batch size of {}'.format(
        args.epochs, args.batch_size))
    if(args.stoch):
        train = train_stoch
    else:
        train = train_non_stoch
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print('Computing on {}'.format('cuda' if use_cuda else 'cpu'))
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if use_cuda else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    if not args.imdb:
        train_loader, test_loader = dataloader.dataloader(args.cifar10, args.fash, args.imdb, device,
                                                          args.batch_size, args.test_batch_size,
                                                          kwargs)
    else:
        train_loader, test_loader, TEXT = dataloader.dataloader(args.cifar10, args.fash, args.imdb, device,
                                                                args.batch_size, args.test_batch_size,
                                                                kwargs)
    # create model
    if args.cifar10:
        model = models.cifar10.GoogLeNet()
        print('Using GoogLeNet model')
    elif not args.imdb:
        model = models.mnist.Net()
        print('Using MNIST model')
    elif args.imdb:
        model = models.imdb.Lstm(vocab_size=len(
            TEXT.vocab), pad_idx=TEXT.vocab.stoi[TEXT.pad_token])
        pretrained_embeddings = TEXT.vocab.vectors

        model.embedding.weight.data.copy_(pretrained_embeddings)
        UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
        PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

        model.embedding.weight.data[UNK_IDX] = torch.zeros(100)
        model.embedding.weight.data[PAD_IDX] = torch.zeros(100)
        print('Using IMDB Model')
    print("Initial model: {}".format(str(model)))
    train_correct = []
    test_correct = []
    train_loss = []
    test_loss = []
    print('Gradient is computed {}stochastically'.format(
        '' if args.stoch else 'non-'))

    model.to(device)
    try:
        extra_params = {}
        if args.optim == 'SGD' and args.momentum != 0.0:
            extra_params = {'momentum': args.momentum, 'nesterov': True}
        if args.optim == 'SSA' or args.optim == 'SSAAda':
            extra_params = {'k': args.k}
        if args.optim == 'SSAAda' or args.optim == 'Adadelta':
            extra_params = {'rho': args.rho, 'eps': args.eps}

        optim_class = getattr(optim, args.optim)
        optimizer = optim_class(model.parameters(), lr=args.lr, **extra_params)
        print(optimizer)
    except Exception as e:
        print(e)
        raise ValueError('Undefined Optimiser: {}'.format(args.optim))

    try:
        if args.loss is None:
            if args.imdb:
                args.loss = 'BCEWithLogitsLoss'
            else:
                args.loss = 'CrossEntropyLoss' if args.cifar10 else 'NLLLoss'
        if 'Loss' not in args.loss:
            raise ValueError('Undefined Loss: {}'.format(args.loss))
        loss_class = getattr(torch.nn, args.loss)
        loss_function = loss_class()
        print('Loss function is: {}'.format(str(loss_function)))

        if args.loss in ['MSELoss']:
            scatter = True
        else:
            scatter = False

    except Exception as e:
        print(e)
        raise ValueError('Undefined Loss: {}'.format(args.loss))

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, loss_function, train_loader, optimizer, epoch, train_correct,
              train_loss, scatter)
        test(args, model, device, test_loader,
             test_correct, test_loss, scatter, loss_function)

    save_result = {'optim': optimizer, 'model': model.state_dict, 'args': args,
                   'train_loss': train_loss, 'test_loss': test_loss,
                   'train_correct': train_correct, 'test_correct': test_correct,
                   'time': time.time() - start_time}

    save_dir = 'results/{}-{}-{}'.format(str(datetime.datetime.now()).replace(' ', '-')
                                         .replace('.', '-').replace(':', '-'), args.optim, args.loss)
    plt, acc_fig, loss_fig = view.gfx(save_result, save_dir)

    save_option = 'y'  # input('Save data? (y)es/(n)o ')

    if save_option == 'y':
        os.makedirs(save_dir)
        save_name = save_dir + '/data.result'
        with open(save_name, 'wb') as f:
            dill.dump(save_result, f)
        plt.savefig(save_dir + '/acc.png', figure=acc_fig)
        plt.savefig(save_dir + '/loss.png', figure=loss_fig)


if __name__ == '__main__':
    main()
