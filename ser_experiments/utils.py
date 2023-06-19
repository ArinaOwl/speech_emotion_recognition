import numpy as np
import tqdm.notebook as tqdm
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import confusion_matrix
import os


def compute_score(conf_matrix, n_classes, classes_dict):
    accuracy = np.zeros(n_classes)
    precision = np.zeros(n_classes)
    recall = np.zeros(n_classes)
    f1 = np.zeros(n_classes)
    for i in range(n_classes):
        tp_fn = np.sum(conf_matrix[i])
        tp_fp = np.sum(conf_matrix[:, i])
        tp = conf_matrix[i, i]

        # accuracy
        tp_tn_fp_fn = np.sum(conf_matrix)
        accuracy[i] = (tp_tn_fp_fn - tp_fn - tp_fp + tp) / tp_tn_fp_fn

        # precision
        if tp_fp == 0:
            precision[i] = 0.0
        else:
            precision[i] = tp / tp_fp

        # recall
        if tp_fn == 0:
            recall[i] = 0.0
        else:
            recall[i] = tp / tp_fn

        # f1
        if tp_fn + tp_fp == 0:
            f1[i] = 0.0
        else:
            f1[i] = 2 * tp / (tp_fn + tp_fp)

    score_dict = {
        "accuracy/macro_average": np.mean(accuracy),
        "precision/macro_average": np.mean(precision),
        "recall/macro_average": np.mean(recall),
        "f1/macro_average": np.mean(f1),
    }

    for i in range(n_classes):
        score_dict["accuracy/{}".format(classes_dict[i])] = accuracy[i]
        score_dict["precision/{}".format(classes_dict[i])] = precision[i]
        score_dict["recall/{}".format(classes_dict[i])] = recall[i]
        score_dict["f1/{}".format(classes_dict[i])] = f1[i]

    return score_dict


def train(model, dataset, loss_func, optimizer, scheduler,
          epoch, batch_size, device, writer,
          classes, checkpoints_dir, gradient_accumulation_steps=1):
    n_classes = len(classes)

    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    pbar = tqdm.tqdm(range(epoch), total=epoch)

    for epoch_idx in pbar:

        dataset.set_split('train')
        dl = DataLoader(dataset=dataset, batch_size=batch_size,
                        shuffle=True, drop_last=True)

        running_loss = 0.0
        running_confusion_matrix = np.zeros((n_classes, n_classes))
        model.train()
        # optimizer.zero_grad()
        for opt in optimizer:
            opt.zero_grad()

        for idx, batch in tqdm.tqdm(enumerate(dl), total=len(dl), leave=False):

            outputs = model(batch['features'].to(device, torch.float32))
            if type(outputs) == tuple:
                outputs, = outputs
            loss = loss_func(outputs, batch['class'].to(device))
            loss_batch = loss.item()
            running_loss += (loss_batch - running_loss) / (idx + 1)

            preds = np.argmax(outputs.cpu().detach().numpy(), axis=1)
            running_confusion_matrix += confusion_matrix(batch['class'], preds,
                                                         labels=np.arange(n_classes))

            loss = loss / gradient_accumulation_steps
            loss.backward()

            if (idx + 1) % gradient_accumulation_steps == 0:
                # optimizer.step()
                # optimizer.zero_grad()
                for opt in optimizer:
                    opt.step()
                    opt.zero_grad()

            outputs.cpu()
            del outputs

        score_dict = compute_score(running_confusion_matrix, n_classes, classes)
        writer.add_scalars('loss', {'train': running_loss}, epoch_idx + 1)
        for key, value in score_dict.items():
            writer.add_scalars(key, {'train': value}, epoch_idx + 1)
        train_loss = running_loss
        train_acc = score_dict["accuracy/macro_average"]

        dataset.set_split('val')
        dl = DataLoader(dataset=dataset, batch_size=batch_size,
                        shuffle=False, drop_last=False)

        running_loss = 0.0
        running_confusion_matrix = np.zeros((n_classes, n_classes))
        model.eval()

        with torch.no_grad():

            for idx, batch in tqdm.tqdm(enumerate(dl), total=len(dl), leave=False):

                outputs = model(batch['features'].to(device, torch.float32))
                if type(outputs) == tuple:
                    outputs, = outputs
                loss = loss_func(outputs, batch['class'].to(device))

                loss_batch = loss.item()
                running_loss += (loss_batch - running_loss) / (idx + 1)

                preds = np.argmax(outputs.cpu().detach().numpy(), axis=1)
                running_confusion_matrix += confusion_matrix(batch['class'], preds,
                                                             labels=np.arange(n_classes))
                outputs.cpu()
                del outputs

        score_dict = compute_score(running_confusion_matrix, n_classes, classes)
        writer.add_scalars('loss', {'validation': running_loss}, epoch_idx + 1)
        for key, value in score_dict.items():
            writer.add_scalars(key, {'validation': value}, epoch_idx + 1)
        val_loss = running_loss
        val_acc = score_dict["accuracy/macro_average"]

        pbar.set_description(
            'Loss (Train/Test): {0:.3f}/{1:.3f}.\nAccuracy (Train/Test): {2:.3f}/{3:.3f}\n'.format(
                train_loss, val_loss, train_acc, val_acc
            )
        )

        # writer.add_scalar('lr', scheduler.get_last_lr()[-1], epoch_idx + 1)
        # scheduler.step()

        for i, lr_scheduler in enumerate(scheduler):
            writer.add_scalar('lr/{}'.format(i), lr_scheduler.get_last_lr()[-1], epoch_idx + 1)
            lr_scheduler.step()

        # writer.add_histogram('cls_token', model.embedding.cls_token, epoch_idx + 1)
        # writer.add_histogram('position_embeddings', model.embedding.position_embeddings, epoch_idx + 1)
        # writer.add_histogram('embedding/linear_1/weight', model.embedding.embeddings[0].weight, epoch_idx + 1)
        # writer.add_histogram('embedding/linear_1/bias', model.embedding.embeddings[0].bias, epoch_idx + 1)
        # writer.add_histogram('embedding/linear_2/weight', model.embedding.embeddings[2].weight, epoch_idx + 1)
        # writer.add_histogram('embedding/linear_2/bias', model.embedding.embeddings[2].bias, epoch_idx + 1)
        # writer.add_histogram('classifier/weight', model.classifier.weight, epoch_idx + 1)
        # writer.add_histogram('classifier/bias', model.classifier.bias, epoch_idx + 1)
        writer.add_histogram('classifier/layernorm/weight', model.classifier.layernorm.weight, epoch_idx + 1)
        writer.add_histogram('classifier/layernorm/bias', model.classifier.layernorm.bias, epoch_idx + 1)
        writer.add_histogram('classifier/dense/weight', model.classifier.dense.weight, epoch_idx + 1)
        writer.add_histogram('classifier/dense/bias', model.classifier.dense.bias, epoch_idx + 1)

        checkpoint = {'state_dict': model.state_dict(),
                      'optimizer': [opt.state_dict() for opt in optimizer],
                      'scheduler': [lr_scheduler.state_dict() for lr_scheduler in scheduler]}
        torch.save(checkpoint, '{0}/checkpoint{1}.pth'.format(checkpoints_dir, epoch_idx + 1))
