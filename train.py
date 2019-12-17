import torch
import torch.nn.functional as F


def train_stoch(args, model, device, loss_function, train_loader, optimizer, epoch, result_correct,
                result_loss, scatter):

    model.train()
    train_loss = 0
    train_correct = 0
    num_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        data, target = batch
        loss = None

        def closure():
            nonlocal data
            nonlocal target
            nonlocal loss
            nonlocal train_loss
            nonlocal train_correct
            nonlocal num_loss

            optimizer.zero_grad()
            if not args.imdb:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                if scatter:
                    target.scatter_(10)

                loss = loss_function(output, target)
                loss.backward()
                train_loss += loss.item()
                num_loss += 1
                pred = output.argmax(dim=1, keepdim=True)
                train_correct += pred.eq(target.view_as(pred)).sum().item()
                return loss
            else:
                optimizer.zero_grad()
                text, text_lengths = batch.text
                predictions = model(text, text_lengths).squeeze(1)
                loss = loss_function(predictions, batch.label)
                num_loss += 1
                loss.backward()

                def correct_preds(preds, y):
                    rounded_preds = torch.round(torch.sigmoid(preds))
                    correct = (rounded_preds == y).float()
                    acc = correct.sum()
                    return acc
                train_correct += correct_preds(predictions, batch.label)

        optimizer.step(closure)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    result_loss.append(train_loss / num_loss)
    result_correct.append(train_correct / len(train_loader.dataset))


def train_non_stoch(args, model, device, loss_function, train_loader, optimizer,
                    epoch, result_correct, result_loss, scatter):
    closure_calls = 0
    train_loss = 0
    num_loss = 0
    train_correct = 0

    def closure():
        nonlocal closure_calls
        nonlocal train_loss
        nonlocal train_correct
        nonlocal num_loss

        closure_calls += 1
        print('\nNumber of closure calls: {}\n'.format(closure_calls))
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            data, target = batch
            if not args.imdb:
                data, target = data.to(device), target.to(device)
                output = model(data)
                if scatter:
                    # TODO scatter
                    target.scatter_(0, torch.Tensor([0]), 10)

                loss = loss_function(output, target)
                loss.backward()
                train_loss += loss.item()
                num_loss += 1
                pred = output.argmax(dim=1, keepdim=True)
                train_correct += pred.eq(target.view_as(pred)).sum().item()
            else:
                text, text_lengths = batch.text
                predictions = model(text, text_lengths).squeeze(1)
                loss = loss_function(predictions, batch.label)
                num_loss += 1
                loss.backward()

                def correct_preds(preds, y):
                    rounded_preds = torch.round(torch.sigmoid(preds))
                    correct = (rounded_preds == y).float()
                    acc = correct.sum()
                    return acc
                train_correct += correct_preds(predictions, batch.label)

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

        return train_loss

    model.train()
    optimizer.step(closure)
    result_loss.append(train_loss / num_loss)
    result_correct.append(train_correct / len(train_loader.dataset))
