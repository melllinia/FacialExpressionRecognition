import torch
import torch.optim as optim
from net import CNN
from emotion_dataset import train_loader, val_loader


def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

model = CNN()
optimizer = optim.Adam(model.parameters())

checkpoint = torch.load('/home/hovhannes/Desktop/FacialExpressionRecognition/model/checkpoints/model.pkl')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epochs']
accuracy_on_validation_set = evaluate(model, val_loader)
accuracy_on_train_set = evaluate(model, train_loader)

print(f'Validation Accuracy: {accuracy_on_validation_set:.2f}%')
print(f'Train Accuracy: {accuracy_on_train_set:.2f}%')