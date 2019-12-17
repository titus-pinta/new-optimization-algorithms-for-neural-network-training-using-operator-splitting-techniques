import torch
import torch.nn.functional as F


def test(args, model, device, test_loader, result_correct, result_loss, scatter, loss_function):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        if args.imdb:
            for batch in test_loader:
                text, text_lengths = batch.text
                predictions = model(text, text_lengths).squeeze(1)
                loss = loss_function(predictions, batch.label)

                def correct_preds(preds, y):
                    rounded_preds = torch.round(torch.sigmoid(preds))
                    correct = (rounded_preds == y).float()
                    acc = correct.sum()
                    return acc
                test_loss += loss.item()
                correct += correct_preds(predictions, batch.label)

        else:
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)

                if scatter:
                    target.scatter_(10)

                # sum up batch loss
                test_loss += loss_function(output,
                                           target, reduction='sum').item()
                # get the index of the max log-probability
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    result_correct.append(correct / len(test_loader.dataset))
    result_loss.append(test_loss)
