from torchvision import datasets, transforms
import torch
import torchtext


def dataloader(cifar10, fash, imdb, device, batch_size, test_batch_size, kwargs):
    if (cifar10 and fash) or (cifar10 and imdb) or (imdb and fash):
        raise ValueError('Please select only one dataset')

    if not cifar10 and not imdb:
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
            batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            dataset(data_path, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=test_batch_size, shuffle=True, **kwargs)

    elif cifar10:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        trainset = datasets.CIFAR10(
            root='./data/cifar10', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                                   **kwargs)
        testset = datasets.CIFAR10(
            root='./data/cifar10', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                                  shuffle=False, **kwargs)

    elif imdb:
        torch.backends.cudnn.deterministic = True

        TEXT = torchtext.data.Field(tokenize='spacy', include_lengths=True)
        LABEL = torchtext.data.LabelField(dtype=torch.float)
        train_data, test_data = torchtext.datasets.IMDB.splits(
            TEXT, LABEL, root='./data/imdb')
        TEXT.build_vocab(train_data, max_size=25000,
                         vectors='glove.6B.100d', unk_init=torch.Tensor.normal_)
        LABEL.build_vocab(train_data)
        train_loader, test_loader = torchtext.data.BucketIterator.splits(
            (train_data, test_data),
            batch_size=batch_size,
            sort_within_batch=True,
            device=device)

    print(train_loader.dataset)

    if imdb:
        return train_loader, test_loader, TEXT
    else:
        return train_loader, test_loader
