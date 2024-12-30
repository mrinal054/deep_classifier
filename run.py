#!/usr/bin/env python
# coding: utf-8
"""
Author: Mrinal Kanti Dhar
October 17, 2024
"""

# * v2: Add albumentations to do augmentation
# * v3_1: Problem fixed for binary classification without onehot
# * v3_1_1: Images are resized but preserving aspect ratios
# * v4: (Obsolete) used to train with customized models
# * v5: Dataloader updated. Now, it can perform - different image manipulations, roi extractions, and create multi-channel images.
# * v6: Run code from config file
# * v7: Tensorboard added with both loss and metrics
# * 701: TTA added
# * 702: Fixed class activation before thresholding
# * 703: K-fold is sent to config file

print('************ The code is loaded ************')

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Append paths
import sys
import os
sys.path.append(os.getcwd() + '/networks/') 
sys.path.append(os.getcwd() + '/utils/') 
sys.path.append(os.getcwd() + '/dataloader/') 
sys.path.append(os.getcwd() + '/losses/') 
sys.path.append(os.getcwd() + '/config/') 

import cv2
from dataloader.loader import MyDataset
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from datetime import datetime
from copy import deepcopy
import matplotlib.pyplot as plt
import pandas as pd
import time
import seaborn as sns

import albumentations as A

from networks import nets
from network_parameters.params import model_params
from utils import augmentations, normalization
from losses.loss import loss_func

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from sklearn.metrics import roc_curve, RocCurveDisplay, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from box import Box
import yaml
from tqdm import tqdm
import argparse

# ### Function to read config file from command line
def get_config_from_args():
    parser = argparse.ArgumentParser(description="Pass config file")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML config file")
    args = parser.parse_args()
    return args
    
# ### Read config file
# Get the config file from command-line arguments
args = get_config_from_args()

with open(args.config, "r") as file:
    config = yaml.safe_load(file)
config = Box(config)

# ### Parameters
#%% Parameters
BASE_MODEL = config.model.name

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = config.train.epochs

LR = config.train.lr #0.0001 # learning rate
WEIGHT_DECAY = config.train.weight_decay #1e-5

SAVE_WEIGHTS_ONLY = config.train.save_weights_only
SAVE_BEST_MODEL = config.train.save_best_model
SAVE_LAST_MODEL = config.train.save_last_model
SAVE_INIT_MODEL = config.train.save_initial_model # useful in cross-validation, save a copy of the base model only
PERIOD = config.train.period # periodically save checkpoints
EARLY_STOP = config.train.early_stop
PATIENCE = config.train.patience # for early stopping

BATCH_SIZE = config.train.batch_size
ONE_HOT = config.train.one_hot
N_CLASSES = config.train.n_classes
ONLY_ADNEXAL = config.data.only_adnexal
ONLY_FLUID = config.data.only_fluid
ONLY_SOLID = config.data.only_solid
DRAW_BBOX = config.data.draw_bbox
CROP_ROI = config.data.crop_roi
MARGIN = config.data.margin
RESIZE = config.data.resize
KEEP_ASPECT_RATIO = config.data.keep_aspect_ratio
TARGET_SIZE = config.data.target_size
CONCAT = config.data.concat # Possible keywords are: "image", "adnexal", "fluid", "solid", "mask"
INPUT_CH = len(CONCAT)

# Parameters for ensemble models
DROPOUT = config.model.dropout
OUT_CHS = config.model.out_channels # concat feature maps will be converted to OUT_CHS

# ### Image directory
root = config.directories.root
result_dir = config.directories.result_dir
train_im_dir = config.directories.train_im_dir
val_im_dir = config.directories.val_im_dir
test_im_dir = config.directories.test_im_dir


# ### Transforms
# Normalization
normalize_transform = normalization.normalize(config) # always normalize

# Augmentation
transform = augmentations.transforms()

# ### Base model name
base_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
if BASE_MODEL == "base_models" or BASE_MODEL == "BaseModelSepIn":
    base_model_name = BASE_MODEL + '_' + config.model.subname + '_' + base_timestamp # general name for all k-fold models
    print("Base model name:", base_model_name)
else:
    base_model_name = BASE_MODEL + '_' + base_timestamp # general name for all k-fold models
    print("Base model name:", base_model_name)


# ### Prepare preprocessing dictionary
pp_dict = {}
pp_dict["only_adnexal"] = ONLY_ADNEXAL
pp_dict["only_fluid"] = ONLY_FLUID
pp_dict["only_solid"] = ONLY_SOLID
pp_dict["draw_bbox"] = DRAW_BBOX
pp_dict["crop_roi"] = CROP_ROI
pp_dict["margin"] = MARGIN
pp_dict["resize"] = RESIZE
pp_dict["keep_aspect_ratio"] = KEEP_ASPECT_RATIO
pp_dict["target_size"] = TARGET_SIZE

# For training data
train_pp_dict = pp_dict.copy()
train_pp_dict["file_dir"] = train_im_dir

# For validation data
val_pp_dict = pp_dict.copy()
val_pp_dict["file_dir"] = val_im_dir

# For test data
test_pp_dict = pp_dict.copy()
test_pp_dict["file_dir"] = test_im_dir


# ### Read adnexal_dataset.xlsx
# Read adnexal dataset.xlsx
df_location = config.directories.excel_dir
dataframe = pd.read_excel(df_location, sheet_name=None) 

# Read train and test sheets
df_train = dataframe['train']  
df_test = dataframe['test']  

print(df_train.head())

train_names = df_train['Base names']
train_class = df_train['Class']

# ### Helper function
#%% Helper function: Save model
def save(model_path, epoch, model_state_dict, optimizer_state_dict):

    state = {
        'epoch': epoch + 1,
        'state_dict': deepcopy(model_state_dict),
        'optimizer': deepcopy(optimizer_state_dict),
        }

    torch.save(state, model_path)


# ### Base model
# Dynamically get the model class from nets
get_model = getattr(nets, config.model.name) # get_model is a model class, not an object

params = model_params(config.model.name, config) # initialize the model with other parameters 

base_model = get_model(**params)

# print(base_model)  # To verify it's working


# ### Train

if config.phase == "train" or config.phase == "both":

    # Initialize summary writer for tensorboard
    writter_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    writer_dir = os.path.join(result_dir, base_model_name, 'logs')
    writer = SummaryWriter(os.path.join(writer_dir, 'fashion_trainer_{}'.format(writter_timestamp)))

    # Save the preprocessing dictionary
    df_pp_dict = pd.DataFrame(list(pp_dict.items()), columns=["Parameter", "Value"]) # convert pp_dict to a DataFrame for saving
    pp_save_dir = os.path.join(result_dir, base_model_name)
    os.makedirs(pp_save_dir, exist_ok=True)
    df_pp_dict.to_excel(os.path.join(pp_save_dir, 'pp_dict_' + base_model_name + '.xlsx'), index=False)
    
    # Save the config file
    with open(os.path.join(pp_save_dir, "config.yaml"), "w") as file:
      yaml.dump(config.to_dict(), file, default_flow_style=False)

    # Save initial model
    if SAVE_INIT_MODEL: torch.save(base_model, os.path.join(pp_save_dir, "initial_model.pth"))
    
    
    ### Uncomment to retrain
    
    # retrain_model_name = 'name_of_the_save_model'
    
    # retrain_checkpoint_loc = '/research/m324371/Project/adnexal/checkpoints/' + retrain_model_name
    # retrain_checkpoint = torch.load(os.path.join(retrain_checkpoint_loc, 'best_model.pth'))
    # model.load_state_dict(retrain_checkpoint['state_dict'])
    # optimizer.load_state_dict(retrain_checkpoint['optimizer'])
    
    ### Training
    
    # ============================================================================ #
    # ============================= Train one epoch ============================== #
    # ============================================================================ #
    
    # Link: https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
    
    # show_report = 50 # reports loss after every 'show_report' batches in an epoch
    
    # def train_one_epoch(epoch_index, tb_writer):
    def train_one_epoch(epoch_index):
        running_loss = 0.
        valid_batches = 0
        train_gt, train_pred, train_prob = [], [], []
    
        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, (inputs, labels, _) in enumerate(train_loader): # >>>>>>>>>>>>> training loader returns input, label, and image name 
            # Every data instance is an input + label pair
    
            # # Skip this iteration if this batch contains None
            # if None in (inputs, labels):
            #   continue
    
            valid_batches += 1
            
            # Moving to GPU
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
    
            # Zero your gradients for every batch!
            optimizer.zero_grad()
    
            # Make predictions for this batch
            outputs = model(inputs)
            
            # Compute the loss and its gradients        
            loss = loss_fn(outputs, labels)
            loss.backward()
    
            # Adjust learning weights
            optimizer.step()
    
            # Gather data and report
            running_loss += loss.item()

            # Collect predictions and probabilities
            prob = torch.softmax(outputs, dim=1) if ONE_HOT else torch.sigmoid(outputs)
            if ONE_HOT:
                pred = torch.argmax(prob, dim=1).cpu().numpy().tolist() # making prediction from probability
                lbls = torch.argmax(labels, dim=1).cpu().numpy().tolist()
                prob_ = prob[:, 1].detach().clone().cpu().numpy().tolist() #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< check if for binary classification
            else:
                pred = (prob > 0.5).float().cpu().numpy().tolist() # making prediction from probability
                lbls = labels.cpu().numpy().tolist()
                prob_ = prob.detach().clone().cpu().numpy().tolist()
    
            train_gt.extend(lbls)
            train_pred.extend(pred)
            train_prob.extend(prob_)
        
            # if i % show_report == show_report-1:
            #     # For loss
            #     last_loss = running_loss / show_report # loss per batch
            #     # print('  batch {} loss: {}'.format(i + 1, last_loss))
            #     tb_x = epoch_index * len(train_loader) + i + 1
            #     tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            #     running_loss = 0.
        
        avg_loss = running_loss / valid_batches if valid_batches > 0 else float('inf')

        # Calculate metrics for the training phase
        train_accuracy = accuracy_score(train_gt, train_pred)
        train_precision = precision_score(train_gt, train_pred)
        train_recall = recall_score(train_gt, train_pred)
        train_f1 = f1_score(train_gt, train_pred)
        
        train_auc = roc_auc_score(train_gt, train_prob) # if len(set(train_gt)) > 1 else 0.0  # avoid AUC error in single-class cases
        
        return avg_loss, train_accuracy, train_precision, train_recall, train_f1, train_auc
    
    
    
    # Create StratifiedKFold object
    list_for_val_result_df = [] # it will store the best results for validation
    list_for_val_preds_df = [] # it will store all validation predictions and probabilities
    # list_for_val_cr_df = []
    
    k = config.train.kfold  # Number of folds
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    # Subplots for training and validation loss
    fig, axs = plt.subplots(1, k, figsize=(18, 4))
    
    # Loop through the folds
    for fold, (train_index, val_index) in enumerate(skf.split(train_names, train_class)):
    
        # # Run for only one fold
        # if fold != 0:
        #     continue
            
        print(f"Fold {fold + 1}")
        print('-' * 40)
        
        print('No. of training images:', len(train_index))
        print('No. of validation images:', len(val_index))
    
        # print('Train index:', train_index)
        # print('Val index:', val_index)
              
        # Create dataframes for this k-fold
        kfold_df_train = df_train.iloc[train_index]
        kfold_df_val = df_train.iloc[val_index]
    
        # Reset index. Start from 0
        kfold_df_train.reset_index(drop=True, inplace=True)
        kfold_df_val.reset_index(drop=True, inplace=True)
    
        #%% Create model name
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        model_name = BASE_MODEL + '_' + timestamp
        print(model_name)
    
        "Create directories"
        val_result_save_dir = os.path.join(result_dir, base_model_name, 'results_val')
        save_fig_dir = os.path.join(result_dir, base_model_name, "plots")
    
        os.makedirs(val_result_save_dir, exist_ok=True)
        os.makedirs(writer_dir, exist_ok=True)
        os.makedirs(save_fig_dir, exist_ok=True)
        
        "Load model"
        # Make a deep copy of the base model
        model = deepcopy(base_model)
        
        model = model.to(DEVICE)
    
        "Optimizer and learning rate scheduler"
        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                      factor=0.1,
                                      mode='min',
                                      patience=10,
                                      min_lr=0.00001,
                                      verbose=True,
                                      )
        
        "Loss function"
        loss_fn = loss_func(config.loss.name)
    
        "Create checkpoint directory"
        checkpoint_loc = os.path.join(result_dir, base_model_name, 'checkpoints', model_name)
        os.makedirs(checkpoint_loc, exist_ok=True)   
        
        "Dataloader"
        train_dataset = MyDataset(
            kfold_df_train, 
            n_classes=N_CLASSES, 
            transform=transform, 
            normalize=normalize_transform,
            one_hot=ONE_HOT,
            preprocess_dict=train_pp_dict,
            concat=CONCAT,
            )
    
        val_dataset = MyDataset(
            kfold_df_val, 
            n_classes=N_CLASSES, 
            transform=None, 
            normalize=normalize_transform,
            one_hot=ONE_HOT,
            preprocess_dict=val_pp_dict,
            concat=CONCAT,
            )
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)    
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
    
        "Containers for validation"
        val_gt, val_pred, val_names, val_prob = [], [], [], []
        
        "Train N epochs"
        # ============================================================================ #
        # ============================== Train N epochs ============================== #
        # ============================================================================ #
        start = time.time() # start of training
    
        best_vloss = 1_000_000.
        best_val_accuracy = 0.
        best_val_precision = 0.
        best_val_recall = 0.
        best_val_f1 = 0.
        best_val_auc = 0.
        save_model = False # initially it is False
        cnt_patience = 0
        initial_epoch = 0
        
        store_train_loss, store_val_loss = [], []
        store_epochs = []
        
        for epoch in range(initial_epoch, EPOCHS):
            print('EPOCH {}:'.format(epoch + 1))
        
            # Training phase
            # Make sure gradient tracking is on, and do a pass over the data
            model.train()
            # avg_loss = train_one_epoch(epoch, writer)
            avg_loss, train_accuracy, train_precision, train_recall, train_f1, train_auc = train_one_epoch(epoch) ############### turned off writer
        
            store_train_loss.append(avg_loss) # average loss is not a tensor
        
            # Validation phase
            # We don't need gradients on to do reporting
            model.eval()

            running_val_gt, running_val_pred, running_val_prob = [], [], []
            with torch.no_grad():
              running_vloss = 0.0
        
              for i, (vinputs, vlabels, vnames) in enumerate(val_loader):
        
                  # Move to GPU
                  vinputs = vinputs.to(DEVICE)
                  vlabels = vlabels.to(DEVICE)
        
                  voutputs = model(vinputs)
                  vloss = loss_fn(voutputs, vlabels)
                  running_vloss += vloss
    
                  # Probabilities 
                  vprob = torch.softmax(voutputs, dim=1) if ONE_HOT else torch.sigmoid(voutputs)
    
                  # Hard prediction (meaning 0 or 1)
                  if ONE_HOT:
                      vout = torch.argmax(vprob, dim=1).data.cpu().numpy().tolist() # making prediction from probability
                      vlbls = torch.argmax(vlabels, dim=1).data.cpu().numpy().tolist()
                      vprob = vprob[:, 1].data.cpu().numpy().tolist() #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< attention: for 2 classes
                  else:
                      vout = (vprob > 0.5).float().data.cpu().numpy().tolist() # **************************** .squeeze()
                      vlbls = vlabels.data.cpu().numpy().tolist()
                      vprob = vprob.data.cpu().numpy().tolist()
    
                  val_gt.extend(vlbls)
                  val_pred.extend(vout)
                  val_prob.extend(vprob)
                  val_names.extend(vnames)
                  store_epochs.extend([epoch]*len(vnames))

                  running_val_gt.extend(vlbls)
                  running_val_pred.extend(vout)
                  running_val_prob.extend(vprob)
        
            avg_vloss = running_vloss / (i + 1)
        
            store_val_loss.append(avg_vloss.data.cpu().numpy().tolist()) # avg_vloss is a tensor and in gpu
        
            # Calculate metrics for the running epoch
            running_val_accuracy = accuracy_score(running_val_gt, running_val_pred)
            running_val_precision = precision_score(running_val_gt, running_val_pred)
            running_val_recall = recall_score(running_val_gt, running_val_pred)
            running_val_f1 = f1_score(running_val_gt, running_val_pred)
            running_val_auc = roc_auc_score(running_val_gt, running_val_prob)

            print(f"Loss train {avg_loss} valid {avg_vloss}, vAccuracy: {running_val_accuracy}, vPrecision: {running_val_precision}, vRecall: {running_val_recall}, vF1: {running_val_f1}, vAUC: {running_val_auc}") 
            
            "Log metrics to TensorBoard"
            writer.add_scalars(f'Loss/Fold{fold+1}', {'training': avg_loss, 'validation': avg_vloss}, epoch + 1)
            writer.add_scalars(f'Metrics/Accuracy/Fold{fold+1}', {'training': train_accuracy, 'validation': running_val_accuracy}, epoch + 1)
            writer.add_scalars(f'Metrics/Precision/Fold{fold+1}', {'training': train_precision, 'validation': running_val_precision}, epoch + 1)
            writer.add_scalars(f'Metrics/Recall/Fold{fold+1}', {'training': train_recall, 'validation': running_val_recall}, epoch + 1)
            writer.add_scalars(f'Metrics/F1_Score/Fold{fold+1}', {'training': train_f1, 'validation': running_val_f1}, epoch + 1)
            writer.add_scalars(f'Metrics/AUC/Fold{fold+1}', {'training': train_auc, 'validation': running_val_auc}, epoch + 1)

            writer.flush()
        
            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                print(f'Validation loss reduced. Saving the model at epoch: {epoch:04d}')
                cnt_patience = 0 # reset patience
                best_model_epoch = epoch
                save_model = True
                # Store metrics for the best model
                val_accuracy = running_val_accuracy
                val_precision = running_val_precision
                val_recall = running_val_recall
                val_f1 = running_val_f1
                val_auc = running_val_auc
                
            else: cnt_patience += 1
        
            # Learning rate scheduler
            scheduler.step(avg_vloss) # monitor validation loss
        
            # Save the model
            if save_model:
                save(os.path.join(checkpoint_loc, 'best_model' + '.pth'),
                      epoch+1, model.state_dict(), optimizer.state_dict())
                save_model = False
        
            # Early stopping
            if EARLY_STOP and cnt_patience >= PATIENCE:
              print(f"Early stopping at epoch: {epoch:04d}")
              break
        
            # Periodic checkpoint save
            if not SAVE_BEST_MODEL:
              if (epoch+1) % PERIOD == 0:
                save(os.path.join(checkpoint_loc, f"cp-{epoch+1:04d}.pth"),
                      epoch+1, model.state_dict(), optimizer.state_dict())
                print(f'Checkpoint saved for epoch {epoch:04d}')
        
            # epoch += 1
        
        if not EARLY_STOP and SAVE_LAST_MODEL:
            print('Saving last model')
            save(os.path.join(checkpoint_loc, 'last_model' + '.pth'),
                  epoch+1, model.state_dict(), optimizer.state_dict())
    
        print('Best model epoch:', best_model_epoch)
        print('Min validation loss:', np.min(store_val_loss))
        end = time.time() # End of training
        exe_time = end - start
        print(f'Training time: {exe_time:.2f} seconds')
        
        "Plot"
        axs[fold].plot(store_train_loss, 'r')
        axs[fold].plot(store_val_loss, 'b')
        axs[fold].set_title("Loss curve")
        axs[fold].legend(['training', 'validation'])    
        plt.tight_layout()
        
        "Validation evaluation"     
        # # Confusion matrix and classification report
        # cm = confusion_matrix(val_gt, val_pred) # confusion matrix    
        # cr = classification_report(val_gt, val_pred, labels=[0, 1], output_dict=True) # classification report    
        # val_cm_df = pd.DataFrame(cr).transpose() # convert confusion matrix to dataframe    
        # list_for_val_cr_df.append(val_cm_df)
    
        # # Precision, recall, F1, accuracy
        # precision = precision_score(val_gt, val_pred)
        # recall = recall_score(val_gt, val_pred)
        # f1 = f1_score(val_gt, val_pred)
        # accuracy = accuracy_score(val_gt, val_pred)
    
        # # AUC
        # auc = roc_auc_score(val_gt, val_prob)
    
        # print("vPrecision:", precision, "vRecall:", recall, "vF1:", f1, "vAccuracy:", accuracy, "vAUC:", auc)
    
        "Store in excel"
        val_results_df = pd.DataFrame(
                    {
                        "Model name": [model_name],
                        "Accuracy": val_accuracy,
                        "Precision": val_precision,
                        "Recall": val_recall,
                        "F1 Score": val_f1,
                        "AUC": val_auc,
                        "Best epoch": best_model_epoch,
                        "Min train loss": np.min(store_train_loss),
                        "Min val loss": np.min(store_val_loss), # or [best_vloss.data.cpu().numpy()]
                        "Optimizer": "Adam",
                        "Scheduler": "ReduceLROnPlateau",
                        "Time": exe_time,
                     }
                )

        val_preds_df = pd.DataFrame(
            {
                "Model name": model_name,
                "Epoch": store_epochs,
                "Names": val_names,
                "Label": val_gt,
                "Prediction": val_pred,
                "Probability": val_prob,
            }
        )
    
        list_for_val_result_df.append(val_results_df)  
        list_for_val_preds_df.append(val_preds_df)    
    
    # fold = 0 # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< reset fold so that for loops run only one iteration
    
    "Save dataframes in excel file"
    with pd.ExcelWriter(os.path.join(val_result_save_dir, base_model_name + "_val.xlsx"), engine='xlsxwriter') as xl_writer:
        for idx, vdf in enumerate(list_for_val_result_df):
            vdf.to_excel(xl_writer, sheet_name=f'fold{idx + 1}', index=False)

    with pd.ExcelWriter(os.path.join(val_result_save_dir, base_model_name + "_preds_val.xlsx"), engine='xlsxwriter') as xl_writer:
        for idx, vdf in enumerate(list_for_val_preds_df):
            vdf.to_excel(xl_writer, sheet_name=f'fold{idx + 1}', index=False)
    
    fig.savefig(os.path.join(save_fig_dir, base_model_name + '.png')) # saving loss curves

    print("*"*20, "Training done", "*"*20)
else:
    print("*"*20, "Skipping training", "*"*20)


# ## Inference
if config.phase == "both" or config.phase == "test":
    from sklearn.metrics import roc_curve, auc
    
    if config.phase == "test":
        base_model_name = config.test.base_model_name
    
    ##### Test dataloader
    
    test_dataset = MyDataset(
            df_test, 
            n_classes=N_CLASSES, 
            transform=None, 
            normalize=normalize_transform,
            one_hot=ONE_HOT,
            preprocess_dict=test_pp_dict,
            concat=CONCAT,
            )
    
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
    
    ##### Test directory
    
    # # Test save directory
    # base_model_name = 'resnet18_imsize256_2024-09-25_16-55-17'
    test_save_dir = os.path.join(result_dir, base_model_name, 'results_test')
    os.makedirs(test_save_dir, exist_ok=True)
    
    ##### Load model
    
    # Find the best model name from the k-fold summary report
    dir_excel = os.path.join(result_dir, base_model_name, 'results_val')
    file_name = base_model_name + '_val.xlsx'
    
    dataframe = pd.read_excel(os.path.join(dir_excel, file_name), sheet_name=None)
    
    metric, folds = [], []
    
    for sheet_name, dframe in dataframe.items():
        first_row = dframe.iloc[0]
        metric.append(first_row["Accuracy"])
        folds.append(sheet_name)
                      
    best_metric_idx = np.argmax(metric)
    best_model_name = dataframe[folds[best_metric_idx]]["Model name"][0]
    best_epoch = dataframe[folds[best_metric_idx]]["Best epoch"][0]

    print('Base model name:', base_model_name)
    print('Best model index:', best_metric_idx)
    print('Best epoch:', best_epoch)
    print('Best model name:', best_model_name)

    if config.test.type == "average" or config.test.type == "both":
        # Get all model names
        model_names = []
        for fold in folds:
            model_name = dataframe[fold]["Model name"][0]
            model_names.append(model_name)
        
        print(model_names)
        
        trained_models = []
        
        for model_name in model_names:
            checkpoint_loc = os.path.join(result_dir, base_model_name, 'checkpoints', model_name)
            checkpoint = torch.load(os.path.join(checkpoint_loc, 'best_model.pth'))
        
            # Make a deep copy of the base model
            model_copy = deepcopy(base_model)
        
            # Load the weights into the copied model
            model_copy.load_state_dict(checkpoint['state_dict'])
            model_copy.eval()  # Set the copied model to evaluation mode
        
            # Append the copied model to the list of trained models
            trained_models.append(model_copy.to(DEVICE))
        
        print('No. of models:', len(trained_models))
        
        
        ##### Evaluation: Avg of all models
        
        
        true, pred, test_name, pred_probs = [], [], [], []
        with torch.no_grad():
            for i, (tinputs, tlabels, tnames) in tqdm(enumerate(test_loader)):
        
                # Move to GPU
                tinputs = tinputs.to(DEVICE)
                tlabels = tlabels.to(DEVICE)
        
                # Prediction
                store_pred = []
                for model in trained_models:
                    model.eval()
                    toutputs_ = model(tinputs)
                    store_pred.append(toutputs_)
                
                # Stack the outputs from each model along a new dimension and take the mean
                # The shape of store_pred will be num_models x batch_size x n_classes
                stacked_preds = torch.stack(store_pred)  # Stack along a new dimension (num_models, batch_size, n_classes)
                toutputs = torch.mean(stacked_preds, dim=0)  # Take the mean across models (batch_size, n_classes)
        
                # Probabilities
                test_prob = torch.softmax(toutputs, dim=1) if ONE_HOT else torch.sigmoid(toutputs)
                
                if ONE_HOT:
                    pred_prob = test_prob[:, 1].data.cpu().numpy().tolist()  # Probability of the positive class (malignant) #<<<<<<<<<<< for 2-cls
                    pred_class = torch.argmax(test_prob, dim=1).data.cpu().numpy().tolist()  # Hard prediction # making prediction from probability
                    tlabels = torch.argmax(tlabels, dim=1).data.cpu().numpy().tolist()
                else:
                    pred_prob = test_prob.data.cpu().numpy().squeeze()  # Probability of the positive class (malignant)
                    pred_class = (test_prob > 0.5).float().data.cpu().numpy().squeeze() # making prediction from probability
                    tlabels = tlabels.data.cpu().numpy().squeeze()
            
                # Collect results
                true.extend(tlabels)
                pred.extend(pred_class)
                pred_probs.extend(pred_prob)
                test_name.extend(tnames)
        
        # Save predictions to an Excel file
        df_test_ = pd.DataFrame(
            {
                "Names": test_name,
                "Label": true,
                "Prediction": pred,
                "Predicted Probability": pred_probs,
            }
        )
        
        df_test_.to_excel(os.path.join(test_save_dir, base_model_name + "_avg.xlsx"))
        
        "Confusion matrix, specificity, precision, recall, F1, accuracy, and AUC"
        y_true = df_test_["Label"].tolist()
        y_pred = df_test_["Prediction"].tolist()
        y_pred_probs = df_test_["Predicted Probability"].tolist()
        
        cm = confusion_matrix(y_true, y_pred) # Confusion matrix
        tn, fp, fn, tp = cm.ravel()  # Unpack confusion matrix values
        
        specificity = tn / (tn + fp)                    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Pay attention: Specificity for binary classification !!!!!!!!!
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        auc_value = roc_auc_score(y_true, y_pred_probs)
        
        print("specificity:", specificity)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("Accuracy:", accuracy)
        print("AUC:", auc_value)
        
        # Save metrics to Excel
        test_results_df = pd.DataFrame(
            {
                "Model name": [model_name],
                "Accuracy": [accuracy],
                "Specificity": [specificity],
                "Precision": [precision],
                "Recall": [recall],
                "F1 Score": [f1],
                "AUC": [auc_value],
            }
        )
        
        test_results_df.to_excel(os.path.join(test_save_dir, base_model_name + "_metrics_avg.xlsx"))
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        cmap = sns.light_palette("green", as_cmap=True)
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, annot_kws={"fontsize": 20}, cbar=False,
                    xticklabels=['Predicted 0', 'Predicted 1'],
                    yticklabels=['Actual 0', 'Actual 1'])
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(test_save_dir, "cmat_" + base_model_name + "_avg.png"))
        
        # Classification report
        cr = classification_report(y_true, y_pred, labels=[0, 1], output_dict=True)
        
        df_cr = pd.DataFrame(cr).transpose()
        df_cr.to_excel(os.path.join(test_save_dir, base_model_name + "_creport_avg.xlsx"))
        
        # Generate ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_probs)  # Use predicted probabilities
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC Curve
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(test_save_dir, "roc_" + base_model_name + "_avg.png"))
        
        print("*"*20, "Avg. model evaluation done", "*"*20)
    
    ##### Evaluation: Best model
    if config.test.type == "best" or config.test.type == "both":
        from sklearn.metrics import roc_curve, auc
        
        checkpoint_loc = os.path.join(result_dir, base_model_name, 'checkpoints', best_model_name)
        checkpoint = torch.load(os.path.join(checkpoint_loc, 'best_model.pth'))

        # Make a deep copy of the base model
        model = deepcopy(base_model)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(DEVICE)
        model.eval()
        
        true, pred, test_name, pred_probs = [], [], [], []
        with torch.no_grad():
            for i, (tinputs, tlabels, tnames) in tqdm(enumerate(test_loader)):
        
                # Move to GPU
                tinputs = tinputs.to(DEVICE)
                tlabels = tlabels.to(DEVICE)
        
                # Prediction 
                toutputs = model(tinputs)
        
                # Probabilities
                test_prob = torch.softmax(toutputs, dim=1) if ONE_HOT else torch.sigmoid(toutputs)
                
                if ONE_HOT:
                    pred_prob = test_prob[:, 1].data.cpu().numpy().tolist()  # Probability of the positive class (malignant) #<<<<<<<<<<< for 2-cls
                    pred_class = torch.argmax(test_prob, dim=1).data.cpu().numpy().tolist()  # Hard prediction
                    tlabels = torch.argmax(tlabels, dim=1).data.cpu().numpy().tolist()
                else:
                    pred_prob = test_prob.data.cpu().numpy().squeeze()  # Probability of the positive class (malignant)
                    pred_class = (test_prob > 0.5).float().data.cpu().numpy().squeeze() # making prediction from probability
                    tlabels = tlabels.data.cpu().numpy().squeeze()
            
                # Collect results
                true.extend(tlabels)
                pred.extend(pred_class)
                pred_probs.extend(pred_prob)
                test_name.extend(tnames)
        
        # Save predictions to an Excel file
        df_test_ = pd.DataFrame(
            {
                "Names": test_name,
                "Label": true,
                "Prediction": pred,
                "Predicted Probability": pred_probs,
            }
        )
        
        df_test_.to_excel(os.path.join(test_save_dir, base_model_name + "_best_model.xlsx"))
        
        "Confusion matrix, specificity, precision, recall, F1, accuracy, and AUC"
        y_true = df_test_["Label"].tolist()
        y_pred = df_test_["Prediction"].tolist()
        y_pred_probs = df_test_["Predicted Probability"].tolist()
        
        cm = confusion_matrix(y_true, y_pred) # Confusion matrix
        tn, fp, fn, tp = cm.ravel()  # Unpack confusion matrix values
        
        specificity = tn / (tn + fp)                      # >>>>>>>>>>>>>>>>>>>> Pay attention: Specificity for binary classification !!!!!!!!!
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        auc_value = roc_auc_score(y_true, y_pred_probs)
        
        print("Specificity:", specificity)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("Accuracy:", accuracy)
        print("AUC:", auc_value)
        
        # Save metrics to Excel
        test_results_df = pd.DataFrame(
            {
                "Model name": [best_model_name],
                "Accuracy": [accuracy],
                "Specificity": [specificity],
                "Precision": [precision],
                "Recall": [recall],
                "F1 Score": [f1],
                "AUC": [auc_value],
            }
        )
        
        test_results_df.to_excel(os.path.join(test_save_dir, base_model_name + '_metrics_best_model.xlsx'))
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        cmap = sns.light_palette("green", as_cmap=True)
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, annot_kws={"fontsize": 20}, cbar=False,
                    xticklabels=['Predicted 0', 'Predicted 1'],
                    yticklabels=['Actual 0', 'Actual 1'])
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(test_save_dir, "cmat_" + base_model_name + "_best_model.png"))
        
        # Classification report
        cr = classification_report(y_true, y_pred, labels=[0, 1], output_dict=True)
        
        df_cr = pd.DataFrame(cr).transpose()
        df_cr.to_excel(os.path.join(test_save_dir, base_model_name + "_creport_best_model.xlsx"))
        
        # Generate ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_probs)  # Use predicted probabilities
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC Curve
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(test_save_dir, "roc_" + base_model_name + "_best_model.png"))
        
        print("*"*20, "Best model evaluation done", "*"*20)

