import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns
#sns.set()


def gfx(save_result, save_name):
    args = save_result['args']
    n = args.epochs
    xaxis = np.linspace(1, n, n)

    if not (args.fash or  args.cifar10):
        dataset = 'MNIST'
    elif args.fash:
        dataset = 'MNISTFashion'
    else:
        dataset = 'Cifar10'

    acc_fig = plt.figure()
    plt.plot(xaxis, save_result['train_correct'], 'rs-', xaxis, save_result['test_correct'], 'bo-')
    plt.legend(['Train Correct', 'Test Correct'])
    plt.title(dataset + ' ' + save_result['args'].optim + ' Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Acccuracy')

    loss_fig = plt.figure()
    plt.plot(xaxis, save_result['train_loss'], 'rs-', xaxis, save_result['test_loss'], 'bo-')
    plt.legend(['Train Loss', 'Test Loss'])
    plt.title(dataset + ' ' + save_result['args'].optim + ' Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    #plt.show()
    return plt, acc_fig, loss_fig
