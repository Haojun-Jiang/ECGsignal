import numpy as np
import torch
import matplotlib.pyplot as plt


def train_module(model, train_loader, val_loader, optimizer, criterion, device, n_epochs, flatten=False):
    model = model.to('cuda')
    
    train_loss_list = []
    val_loss_list = []

    train_acc_list = []
    val_acc_list = []

    y_true_list = []
    y_pred_list = []

    y_true_val_list = []
    y_pred_val_list = []

    y_score_list = []

    best_val_acc = 0.0
    best_model_wts = None

    for epoch in range(n_epochs):
        model.train()
        correct = 0
        total = 0
        running_loss = 0.0
        n = 0
        
        for input, labels in train_loader:
            input, labels= input.to(device), labels.to(device)
            if flatten:
                input = input.view(input.size(0), -1)
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            y_true_list.extend(labels.cpu().numpy())
            y_pred_list.extend(predicted.cpu().numpy())
            
            n+=1
        tr_loss=running_loss/n
        tr_acc = correct / total
        y_true_val_list, y_pred_val_list, y_score_list,val_loss, val_acc = evaluate(model, val_loader, criterion, device, flatten)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = model.state_dict()
            # torch.save(best_model_wts, f'./results/best_{model_name}_wts.pt')

        train_loss_list.append(tr_loss)
        train_acc_list.append(tr_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {tr_loss:.3f} | Val Loss: {val_loss:.3f} | Train Acc: {tr_acc:.1%} | Val Acc: {val_acc:.1%}")

    return {
        'train_loss': train_loss_list,
        'val_loss': val_loss_list,
        'train_acc': train_acc_list,
        'val_acc': val_acc_list,
        'y_true': y_true_list,
        'y_pred': y_pred_list,
        'y_true_val' : y_true_val_list,
        'y_pred_val' : y_pred_val_list,
        'y_score': y_score_list
    }


# evaluate function
def evaluate(model, dataloader ,loss_fn, device ,flatten=False):
    model.eval()
    total = 0
    correct = 0
    loss_total = 0

    y_true_list = []
    y_pred_list = []
    y_score_list = []

    with torch.no_grad():
        for input, labels in dataloader:
            input, labels = input.to(device), labels.to(device)
            if flatten:
                input = input.view(input.size(0),-1)
            output = model(input)
            loss = loss_fn(output, labels)
            loss_total += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            y_true_list.extend(labels.cpu().numpy())
            y_pred_list.extend(predicted.cpu().numpy())
            y_score_list.extend(output.cpu().numpy())

    # return avg_loss and acc
    return y_true_list, y_pred_list, y_score_list, loss_total/len(dataloader), correct/total

# plot function
def training_plots(result_path, model_name):
    results = torch.load(result_path)
    epochs = range(1, len(results['train_loss'])+1)
    plt.figure(figsize = (12, 5))

    plt.subplot(1,2,1)
    plt.plot(epochs, results['train_loss'], label= 'Train loss')
    plt.plot(epochs, results['val_loss'], label='Val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} - Loss over Epochs')
    plt.legend()
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(epochs, results['train_acc'], label= 'Train Accuracy')
    plt.plot(epochs, results['val_acc'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f"{model_name} - Accuracy over Epochs")
    plt.legend()
    plt.grid(True)

    plt.show()

    return 0