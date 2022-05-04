import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import torchvision
import matplotlib.pyplot as plt

from dataset import PerceptionDataset, train_transforms, test_transforms


def get_model_object_detection(num_classes, load_default=True):
    # load a model pre-trained on COCO
    model = torchvision.models.efficientnet_b2(pretrained=load_default)

    for param in model.parameters():
        param.requires_grad = False
    # print(model)
    
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    # fc    classifier[1]    heads.head
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(nn.Linear(num_ftrs, 512),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(512, num_classes),
                                nn.LogSoftmax(dim=1))

    optimizer = optim.Adam(model.classifier[1].parameters(), lr=0.003)    
    return model, optimizer


def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Use our dataset and defined transformations
    dataset_train = PerceptionDataset('data/dataset', train_transforms, train=True)
    dataset_test = PerceptionDataset('data/dataset', test_transforms, train=True)
    # our dataset has two classes only - background and person
    num_classes = len(dataset_train.class_to_idx)

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset_train)).tolist()
    dataset_train = torch.utils.data.Subset(dataset_train, indices[:-16])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-16:])

    # define training and validation data loaders
    train_loader = DataLoader(
        dataset_train, batch_size=128, shuffle=True, num_workers=4)
    test_loader = DataLoader(
        dataset_test, batch_size=16, shuffle=True, num_workers=4)

    # get the model using our helper function
    model, optimizer = get_model_object_detection(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    criterion = nn.NLLLoss()

    # let's train it for 10 epochs
    num_epochs = 150
    steps = 0
    running_loss = 0
    print_every = 20
    train_losses, test_losses = [], []

    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            steps += 1
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        test_loss += batch_loss.item()
                        
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                train_losses.append(running_loss/len(train_loader))
                test_losses.append(test_loss/len(test_loader))                    
                print(f"Epoch {epoch+1}/{num_epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Test loss: {test_loss/len(test_loader):.3f}.. "
                    f"Test accuracy: {accuracy/len(test_loader):.3f}")
                running_loss = 0
                model.train()

    torch.save(model.state_dict(), 'data/models/model_weights.pth')

    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Validation loss')
    plt.legend(frameon=False)
    plt.show()

    print("That's it!")



if __name__ == "__main__":
    main()
