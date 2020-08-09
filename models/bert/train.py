import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from pytorch_transformers import RobertaConfig, RobertaTokenizer, RobertaModel
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from model import MatchArchitecture
from data_utils import Matching_dataset

SEQ_LEN = 10
RNN_DIM = 64
LINEAR_DIM=32
CLASSES = 1
ROBERTA_FEAT_SIZE = 768
F1_POS_THRESHHOLD = .1
epsilon = 1e-8

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

VERSION = '1.1'
SAVE_DIR = 'C:\\Users\\Raymond\\Documents\\Github\\clinical-abbreviations\\models\\checkpoints\\{}.pt'.format(VERSION)
train_data_path = 'C:\\Users\\Raymond\\Documents\\Github\\clinical-abbreviations\\training\\Train1_training.csv'
val_data_path = 'C:\\Users\\Raymond\\Documents\\Github\\clinical-abbreviations\\training\\Train1_val.csv'

train_dataset = Matching_dataset(train_data_path, tokenizer)
val_dataset = Matching_dataset(val_data_path, tokenizer)

model = MatchArchitecture(
    None,
    'roberta-base',
    False,
    None,
    ROBERTA_FEAT_SIZE,
    CLASSES,
    RNN_DIM,
    LINEAR_DIM,
).cuda()

def lr_scheduler(epoch):
    if epoch < 7:
        return 3e-5
    if epoch < 10:
        return 3e-6
    else:
        return 3e-7

train_config = {
    "batch_size": 16,
    "base_lr": .0001,
    "lr_shceduler": lr_scheduler,
    "n_epochs": 20

}


def _run_training_loop(model, train_config):
    """Runs the training loop to train the NER."""
    # set up params for training loop

    criterion = nn.BCELoss(reduce=False)
    opt = Adam(model.parameters(), lr=train_config["base_lr"])

    epoch_learn_rates = []
    epoch_train_losses = []
    epoch_train_f1s = []
    epoch_validation_losses = []
    epoch_validation_f1s= []
    train_steps_per_epoch = int(len(train_dataset) / train_config["batch_size"])
    validation_steps_per_epoch = int(len(val_dataset) / train_config["batch_size"])

    for epoch in range(train_config["n_epochs"]):

        train_generator = iter(DataLoader(train_dataset, batch_size=train_config["batch_size"], shuffle=True,
                                          num_workers=4))
        val_generator = iter(DataLoader(train_dataset, batch_size=train_config["batch_size"], shuffle=False,
                                        num_workers=4))

        adjusted_lr = lr_scheduler(epoch)
        for param_group in opt.param_groups:
            param_group["lr"] = adjusted_lr
            epoch_learn_rates.append(adjusted_lr)

        print("Epoch: {}. LR: {}.".format(epoch, adjusted_lr))


        model.train(True)
        running_train_loss = 0
        target_true = 0
        predicted_true = 0
        correct_true_preds = 0
        mask_sum = 0
        y_sum = 0
        for step in range(train_steps_per_epoch):
            # Calculate losses

            sample = next(train_generator)
            X_batch_1 = sample['text_1'].cuda()
            X_batch_2 = sample['text_2'].cuda()
            y_batch = sample['labels'].cuda()
            
            y_sum += torch.sum(y_batch).item() / train_config["batch_size"]
            model.zero_grad()
            sigmoid_output = model(X_batch_1, X_batch_2)
            sigmoid_output = torch.squeeze(sigmoid_output, dim=-1)

            loss = criterion(sigmoid_output, y_batch)
            loss = torch.mean(loss)

            y_batch = y_batch.cpu()

            # Calculate metrics
            running_train_loss += loss.cpu().item()

            threshold_output = (sigmoid_output > F1_POS_THRESHHOLD).cpu().type(torch.IntTensor)
            target_true += torch.sum(y_batch == 1).float().item()
            predicted_true += torch.sum(threshold_output).float().item()
            correct_true_preds += torch.sum(
                ((threshold_output == y_batch) * threshold_output)
                == 1).cpu().float().item()

            # Propogate
            loss.backward()
            opt.step()

            if step % 50 == 0:
                print("train step: ", step, "loss: ", running_train_loss/(step + 1))
                print("mask sum: ", mask_sum/(step + 1), "y_sum: ", y_sum/(step + 1))

            del loss, X_batch_1, X_batch_2, y_batch, sample, sigmoid_output, threshold_output

        recall = correct_true_preds / (target_true + .1)
        precision = correct_true_preds / (predicted_true +.1)
        epoch_train_f1 = 2 * (precision * recall) / (precision + recall + epsilon)
        epoch_train_f1s.append(epoch_train_f1)
        epoch_train_loss = running_train_loss / train_steps_per_epoch
        epoch_train_losses.append(epoch_train_loss)
        print("Epoch {}, train loss of {}.".format(epoch, epoch_train_loss))
        print("Epoch {}, train f1 of {}.".format(epoch, epoch_train_f1))

        model.train(False)
        running_validation_loss = 0
        val_target_true = 0
        val_predicted_true = 0
        val_correct_true_preds = 0
        for step in range(validation_steps_per_epoch):

            sample = next(val_generator)
            X_batch_1 = sample['text_1'].cuda()
            X_batch_2 = sample['text_2'].cuda()
            y_batch = sample['labels'].cuda()
            
            y_sum += torch.sum(y_batch).item() / train_config["batch_size"]
            model.zero_grad()
            sigmoid_output = model(X_batch_1, X_batch_2)
            sigmoid_output = torch.squeeze(sigmoid_output, dim=-1)

            loss = criterion(sigmoid_output, y_batch)
            loss = torch.mean(loss)

            y_batch = y_batch.cpu()
            # Calculate metrics
            running_validation_loss += loss.cpu().item()
            threshold_output = (sigmoid_output > F1_POS_THRESHHOLD).cpu().type(torch.IntTensor)
            val_target_true += torch.sum(y_batch == 1).float().item()
            val_predicted_true += torch.sum(threshold_output).float().item()
            val_correct_true_preds += torch.sum(
                ((threshold_output == y_batch) * threshold_output)
                == 1).cpu().float().item()


            del loss, X_batch_1, X_batch_2, y_batch, sample, sigmoid_output, threshold_output

        recall = val_correct_true_preds / (val_target_true +epsilon)
        precision = val_correct_true_preds / (val_predicted_true+epsilon)
        epoch_validation_f1 = 2 * (precision * recall) / (precision + recall + epsilon)
        epoch_validation_f1s.append(epoch_validation_f1)
        epoch_validation_loss = running_validation_loss / validation_steps_per_epoch
        epoch_validation_losses.append(epoch_validation_loss)
        print("Epoch {}, train loss of {}.".format(epoch, epoch_train_loss))
        print("Epoch {}, train f1 of {}.".format(epoch, epoch_train_f1))
        print("Epoch {}, validation loss of {}.".format(epoch, epoch_validation_loss))
        print("Epoch {}, validation f1 of {}.".format(epoch, epoch_validation_f1))
        

    torch.save(model.state_dict, SAVE_DIR + 'model.pt')

    train_history = {
        "f1": epoch_train_f1s,
        "loss": epoch_train_losses,
        "val_f1": epoch_validation_f1s,
        "val_loss": epoch_validation_losses,
        "lr": epoch_learn_rates,
    }
    return model, train_history

model, train_hstory = _run_training_loop(model, train_config)