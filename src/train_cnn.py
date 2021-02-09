import os
import torch
import torch.nn as nn
import torch.optim as optim

def train_cnn(args, loaders, model, DEVICE):
    train_loader, valid_loader, test_loader = loaders
    model.to(DEVICE)
    LEARN_RATE = 1.5e-3
    EPOCHS = 5
    criterion = nn.CrossEntropyLoss()   # Combines nn.LogSoftmax() and nn.NLLLoss() in one single class.
    optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE)
    model_dir = '../result/fmnist/'
    model_file = model_dir + args.encoder + '.pt'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    acc_best = 0.0
    for epoch in range(EPOCHS):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        acc_train = evaluation(model, train_loader, DEVICE)
        acc_valid = evaluation(model, valid_loader, DEVICE)
        acc_test = evaluation(model, test_loader, DEVICE)
        print('Epoch: %d/%d, Train acc: %0.2f, Valid acc: %0.2f, Test acc: %0.2f' %
              (epoch, EPOCHS, acc_train, acc_valid, acc_test))

        if acc_valid > acc_best:
            acc_best = acc_valid
            torch.save(model.state_dict(), model_file)  # state_dict() saved in a .pt file


def evaluation(model, dataloader, DEVICE):
    total, correct = 0, 0

    # keep the network in evaluation mode
    model.eval()
    for data in dataloader:
        inputs, labels = data
        # move the inputs and labels to gpu
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    return 100 * correct / total
