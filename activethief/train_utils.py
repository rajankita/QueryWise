import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm
import torch
import torch.utils


def classwise_accuracy(true_labels, predicted_labels):
    # Check if the input arrays have the same length
    if len(true_labels) != len(predicted_labels):
        raise ValueError("Input arrays must have the same length.")

    # Create a confusion matrix
    confusion_matrix = np.zeros((3, 3), dtype=int)

    # Populate the confusion matrix
    for true_label, predicted_label in zip(true_labels, predicted_labels):
        confusion_matrix[true_label, predicted_label] += 1

    # Compute classwise accuracy
    classwise_accuracy = np.zeros(3)
    for i in range(3):
        if np.sum(confusion_matrix[i, :]) != 0:
            classwise_accuracy[i] = confusion_matrix[i, i] / np.sum(confusion_matrix[i, :])
            # print(np.sum(confusion_matrix[i, :]))

    return classwise_accuracy


def testz(model, dataloader, verbose=True):
    """
    Evaluate a model on given dataloader. 
    
    Return: 
    accuracy, f1 score, specificity, sensitivity, class-wise accuracy 
    """
    model.eval()

    trues = []
    preds = []
    # print("Calculating metrics")
    with torch.no_grad():
        for data in (dataloader):
            inputs = data[0].cuda()
            labels = data[1].cuda()

            scores = model(inputs)
            _, pred = torch.max(scores.data, 1)

            preds.append(pred.cpu())
            trues.append(labels.cpu())

        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
    
    cac = classwise_accuracy(trues, preds)
    acc = accuracy_score(trues, preds)
    cfm = confusion_matrix(trues, preds)
    spec = (cfm[0][0] + cfm[0][1] + cfm[1][0] + cfm[1][1])/(np.sum(cfm[0]) + np.sum(cfm[1]))
    sens = cfm[2][2]/np.sum(cfm[2])
    f1 = f1_score(y_true=trues, y_pred=preds, average='macro')

    if verbose == True:
        print('specificity = {}/{}'.format(cfm[0][0] + cfm[0][1] + cfm[1][0] + cfm[1][1], np.sum(cfm[0]) + np.sum(cfm[1])))
        print('sensitivity = {}/{}'.format(cfm[2][2], np.sum(cfm[2])))
    
    return acc, f1, spec, sens, cac


def agree(model1, model2, test_loader):
    c=0
    l=0
    model1.eval()
    model2.eval()
    with torch.no_grad():
        for data in test_loader:
            images = data[0]
            targets = data[1]
            images, targets = images.float().cuda(), targets.cuda()
            if len(images.shape) == 5:
                images = images.squeeze(0)
                
                outputs =  model1(images)
                _, pred = torch.max(outputs, dim=1)
                x1 = torch.max(pred)
                
                outputs =  model2(images)
                _, pred = torch.max(outputs, dim=1)
                x2 = torch.max(pred)

                n = images.shape[0]
                c+=n-int((torch.count_nonzero(x1-x2)).detach().cpu())
                l+=n
            else:
                n=images.shape[0]
                x1=model1(images).argmax(axis=-1,keepdims=False)
                x2=model2(images).argmax(axis=-1,keepdims=False)
                c+=n-int((torch.count_nonzero(x1-x2)).detach().cpu())
                l+=n
            # print(c, l)
    # print('Agreement between Copy and source model is ', c/l)
    return c / l


def dist(indices, dataloader):
    "Return label distribution of selected samples" 
    # create dataloader from dataset
    # dl=DataLoader(dz, batch_size=1, sampler=SubsetRandomSampler(indices), pin_memory=False)
    dl = dataloader
    d = {}
    print('Number of samples ', len(indices))
    labels = []
    with torch.no_grad():
        for data in (dl):
            label = data[1]
            # print(data[1].shape)
            # if soft label, extract hard label using argmax
            if len(label.shape) > 1:
                label = label.argmax(axis=1)
            labels.extend(label.cpu().detach().numpy())
    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        d[int(lbl)] = 0
    for label in labels:
        d[int(label)]+=1
    return d


def train_with_validation(model, criterion, optimizers, scheduler, dataloaders, num_epochs, trial, cycle, out_dir, display_every = 5, early_stop_tolerance=100, la=None):
    print('>> Train a Model.')

    exit = False
    best_f1 = None
    no_improvement = 0

    for epoch in tqdm(range(num_epochs), leave=False):

        t_loss = train_epoch(model, criterion, optimizers, dataloaders, la=la)
        scheduler.step()

        if (epoch+1)%2==0:
            val_acc, val_f1, spec, sens, cac = testz(model, dataloaders['val'], verbose=False)
            # test_acc, test_f1 = testz(model, dataloaders['test'])

            if best_f1 is None or val_f1 > best_f1 :
                best_f1 = val_f1
                torch.save({
                    'trial': trial + 1,
                    'cycle': cycle + 1, 
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    },
                    f'{out_dir}/trial_{trial+1}_cycle_{cycle+1}_best.pth')

                # print(f"Epoch {epoch+1} Model saved in path: {os.path.join(out_dir, f'trial_{trial+1}_best.pth')}")
                no_improvement = 0
            else:
                no_improvement += 1
                
                if (no_improvement % early_stop_tolerance) == 0:
                    exit = True

        # Display progress
        if (epoch+1) % display_every == 0:
            train_acc, train_f1, spec1, sens1, cac1 = testz(model, dataloaders['train'], verbose=False)
            test_acc, test_f1, spec2, sens2, cac2 = testz(model, dataloaders['test'], verbose=False)
            print(f"Epoch {epoch+1}: Train acc/f1 = {train_acc:.4f} / {train_f1:.4f} / {spec1:.4f} / {sens1:.4f}/ {cac1} \n\
                Val acc/f1/spec/sens = {val_acc:.4f} / {val_f1:.4f} / {spec:.4f} / {sens:.4f}/ {cac}\n\
                Test acc/f1/spec/sens = {test_acc:.4f} / {test_f1:.4f} / {spec2:.4f} / {sens2:.4f}/ {cac2}")

        if exit:
            print(f"Number of epochs processed: {epoch+1} in cycle {cycle+1}") 
            break

    train_acc, train_f1, spec1, sens1, cac1 = testz(model, dataloaders['train'])
    val_acc, val_f1, spec2, sens2, cac2 = testz(model, dataloaders['val'])
    test_acc, test_f1, spec3, sens3, cac3 = testz(model, dataloaders['test'])

    print(f"Trial {trial+1}, Cycle {cycle+1}")
    print(f"Train acc/f1/spec/sens = {train_acc:.4f} / {train_f1:.4f} / {spec1:.4f} / {sens1:.4f}/ {cac1}")
    print(f"Val acc/f1/spec/sens = {val_acc:.4f} / {val_f1:.4f} / {spec2:.4f} / {sens2:.4f}/ {cac2}")
    print(f"Test acc/f1/spec/sens = {test_acc:.4f} / {test_f1:.4f} / {spec3:.4f} / {sens3:.4f}/ {cac3}")

    # Save the last model
    torch.save({'trial': trial + 1,
                'cycle': cycle + 1, 
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                },
                f'{out_dir}/trial_{trial+1}_cycle_{cycle+1}_last.pth')

    print('>> Finished.')


def train_epoch(model, criterion, optimizer, dataloaders, vis=None, plot_data=None, la=None):
    model.train()
    iters = 0
    total_loss = 0

    for data in dataloaders['train']:
        input = data[0].cuda()
        target = data[1].cuda()
        iters += 1

        optimizer.zero_grad()
        output = model(input)

        if la is not None:
            output = output + la
        target_loss = criterion(output, target)

        loss = torch.sum(target_loss) / target_loss.size(0)
        total_loss += torch.sum(target_loss)

        loss.backward()
        optimizer.step()

        # Visualize
        if (iters % 100 == 0) and (vis != None) and (plot_data != None):
            plot_data['X'].append(iters)
            plot_data['Y'].append(loss.item())
            vis.line(
                X=np.stack([np.array(plot_data['X'])] * len(plot_data['legend']), 1),
                Y=np.array(plot_data['Y']),
                opts={
                    'title': 'Loss over Time',
                    'legend': plot_data['legend'],
                    'xlabel': 'Iterations',
                    'ylabel': 'Loss',
                    'width': 1200,
                    'height': 390,
                },
                win=1
            )

    mean_loss = total_loss / iters
    return mean_loss