import torch
import torch.nn.parallel
import torch.optim
from tqdm import tqdm

from models.layers import *


def train(model, device, train_loader, criterion, optimizer, T, dvs):
    running_loss = 0
    model.train()
    total = 0
    correct = 0
    for i, (images, labels) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        labels = labels.to(device)
        images = images.to(device)
        if dvs:
            images = images.transpose(0, 1)
        if T == 0:
            outputs = model(images)
        else:
            outputs = model(images).mean(0)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        loss.mean().backward()
        optimizer.step()
        total += float(labels.size(0))
        _, predicted = outputs.cpu().max(1)
        correct += float(predicted.eq(labels.cpu()).sum().item())
    return running_loss, 100 * correct / total

def train_poisson(model, device, train_loader, criterion, optimizer, T):
    running_loss = 0
    model.train()
    M = len(train_loader)
    total = 0
    correct = 0
    for i, (images, labels) in enumerate((train_loader)):
        optimizer.zero_grad()
        labels = labels.to(device)
        images = images.to(device)
        if T > 0:
            outputs = model(images).mean(0)
        else:
            outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        loss.mean().backward()
        optimizer.step()
    
        total += float(labels.size(0))
        _, predicted = outputs.cpu().max(1)
        correct += float(predicted.eq(labels.cpu()).sum().item())
    return running_loss, 100 * correct / total

def val(model, test_loader, device, T, dvs, atk=None):
    correct = 0
    total = 0
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader)):
        inputs = inputs.to(device)
        if dvs:
            inputs = inputs.transpose(0, 1)
        if atk is not None:
            atk.set_training_mode(model_training=False, batchnorm_training=False, dropout_training=False)
            inputs = atk(inputs, targets.to(device))
            model.set_simulation_time(T)
        with torch.no_grad():
            if T > 0:
                outputs = model(inputs).mean(0)
            else:
                outputs = model(inputs)
        _, predicted = outputs.cpu().max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())
    final_acc = 100 * correct / total
    return final_acc

def val_success_rate(model, test_loader, device, T, dvs, atk=None):
    correct = 0
    total = 0
    tt = 0
    model.eval()
    for batch_idx, (inputs, targets) in enumerate((test_loader)):
        inputs = inputs.to(device)
        if dvs:
            inputs = inputs.transpose(0, 1)
        with torch.no_grad():
            if T > 0:
                outputs = model(inputs).mean(0)
            else:
                outputs = model(inputs)
        _, predicted = outputs.cpu().max(1)
        mask = predicted.eq(targets).float()
        
        if atk is not None:
            atk.set_training_mode(model_training=False, batchnorm_training=False, dropout_training=False)
            inputs = atk(inputs, targets.to(device))
            model.set_simulation_time(T)
            
        with torch.no_grad():
            if T > 0:
                outputs = model(inputs).mean(0)
            else:
                outputs = model(inputs)
        _, predicted = outputs.cpu().max(1)
        
        predicted = ~(predicted.eq(targets))
        total += mask.sum()
        correct += (predicted.float()*mask).sum()

    final_acc = 100 * correct / total
    return final_acc