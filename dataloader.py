from torchvision import datasets, transforms
import torch

def dataloader(cifar10, fash, batch_size, test_batch_size, kwargs):

    if cifar10 and fash:
        raise ValueError('Please select only one dataset')

    if not cifar10:
        if fash:
            data_path = './data/fash'
            dataset = datasets.FashionMNIST

        else:
            data_path = './data/mnist'
            dataset = datasets.MNIST

        train_loader = torch.utils.data.DataLoader(
            dataset(data_path, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=batch_size, shuffle=True, **kwargs )
        test_loader = torch.utils.data.DataLoader(
            dataset(data_path, train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=test_batch_size, shuffle=True, **kwargs)


    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                                   **kwargs)
        testset = datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                                  shuffle=False, **kwargs)


    print(train_loader.dataset)
    return train_loader, test_loader
