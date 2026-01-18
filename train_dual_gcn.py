import torch
import torch.optim as optim
import torch.nn.functional as F

import time
import random
import os
import sys

from config import *
from volleyball import *
from collective import *
from dataset import *
from dual_gcn_model import *
from utils import *


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
            
def adjust_lr(optimizer, new_lr):
    print('change learning rate:',new_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def train_net(cfg):
    """
    Training dual GCN net with appearance and motion streams
    """
    os.environ['CUDA_VISIBLE_DEVICES']=cfg.device_list
    
    # Show config parameters
    cfg.init_config()
    show_config(cfg)
    
    # Reading dataset
    training_set,validation_set=return_dataset(cfg)
    
    params = {
        'batch_size': cfg.batch_size,
        'shuffle': True,
        'num_workers': 4
    }
    training_loader=data.DataLoader(training_set,**params)
    
    params['batch_size']=cfg.test_batch_size
    validation_loader=data.DataLoader(validation_set,**params)
    
    # Set random seed
    np.random.seed(cfg.train_random_seed)
    torch.manual_seed(cfg.train_random_seed)
    random.seed(cfg.train_random_seed)

    # Set data position
    if cfg.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Build model and optimizer
    model = DualGCN_Model(cfg)
    
    # Load stage1 model if specified (backbone weights)
    if hasattr(cfg, 'stage1_model_path') and cfg.stage1_model_path:
        model.loadmodel(cfg.stage1_model_path)
    
    if cfg.use_multi_gpu:
        model=nn.DataParallel(model)

    model=model.to(device=device)
    
    model.train()
    model.apply(set_bn_eval)
    
    optimizer=optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=cfg.train_learning_rate,weight_decay=cfg.weight_decay)

    if cfg.test_before_train:
        test_info=test(validation_loader, model, device, 0, cfg)
        print(test_info)

    # Training iteration
    best_result={'epoch':0, 'activities_acc':0}
    start_epoch=1
    for epoch in range(start_epoch, start_epoch+cfg.max_epoch):
        
        if epoch in cfg.lr_plan:
            adjust_lr(optimizer, cfg.lr_plan[epoch])
            
        # One epoch of forward and backward
        train_info=train(training_loader, model, device, optimizer, epoch, cfg)
        show_epoch_info('Train', cfg.log_path, train_info)

        # Test
        if epoch % cfg.test_interval_epoch == 0:
            test_info=test(validation_loader, model, device, epoch, cfg)
            show_epoch_info('Test', cfg.log_path, test_info)
            
            if test_info['activities_acc']>best_result['activities_acc']:
                best_result=test_info
            print_log(cfg.log_path, 
                      'Best group activity accuracy: %.2f%% at epoch #%d.'%(best_result['activities_acc'], best_result['epoch']))
            
            # Save model
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            filepath=cfg.result_path+'/dual_gcn_epoch%d_%.2f%%.pth'%(epoch,test_info['activities_acc'])
            torch.save(state, filepath)
            print('model saved to:',filepath)   
   
def train(data_loader, model, device, optimizer, epoch, cfg):
    """
    Training function with BCE loss for multi-label classification
    """
    activities_meter=AverageMeter()
    loss_meter=AverageMeter()
    epoch_timer=Timer()
    
    for batch_data in data_loader:
        model.train()
        model.apply(set_bn_eval)
    
        # prepare batch data
        batch_data=[b.to(device=device) for b in batch_data]
        batch_size=batch_data[0].shape[0]
        
        # Extract labels - for multi-label, activities should be [B, num_activities] with binary labels
        if len(batch_data) == 5:
            images_in, boxes_in, actions_in, activities_in, bboxes_num_in = batch_data[0], batch_data[1], batch_data[2], batch_data[3], batch_data[4]
        else:
            images_in, boxes_in, actions_in, activities_in = batch_data[0], batch_data[1], batch_data[2], batch_data[3]
            bboxes_num_in = None
        
        # For multi-label: convert single labels to binary labels
        # activities_in: [B, T] or [B] - single class labels
        # Convert to [B, num_activities] binary format
        if len(activities_in.shape) == 2:
            activities_in = activities_in[:, 0]  # Take first frame label [B]
        
        # Create binary labels [B, num_activities]
        num_activities = cfg.num_activities
        activities_binary = torch.zeros(batch_size, num_activities, device=device)
        activities_binary.scatter_(1, activities_in.unsqueeze(1), 1)  # One-hot encoding
        
        # Forward pass
        if bboxes_num_in is not None:
            activities_scores = model((images_in, boxes_in, bboxes_num_in))
        else:
            activities_scores = model((images_in, boxes_in))
        
        # BCE loss for multi-label classification
        # activities_scores: [B, num_activities] (logits)
        # activities_binary: [B, num_activities] (binary labels)
        activities_loss = F.binary_cross_entropy_with_logits(activities_scores, activities_binary)
        
        # Calculate accuracy (for multi-label: average over classes)
        activities_probs = torch.sigmoid(activities_scores)  # [B, num_activities]
        activities_pred = (activities_probs > 0.5).float()  # [B, num_activities]
        
        # Compute accuracy: intersection over union or average precision
        # Simple approach: average per-sample accuracy
        correct = (activities_pred == activities_binary).float()
        activities_accuracy = correct.mean(dim=1).mean().item()  # Average over samples
        
        activities_meter.update(activities_accuracy, batch_size)
        loss_meter.update(activities_loss.item(), batch_size)

        # Optimize
        optimizer.zero_grad()
        activities_loss.backward()
        optimizer.step()
    
    train_info={
        'time':epoch_timer.timeit(),
        'epoch':epoch,
        'loss':loss_meter.avg,
        'activities_acc':activities_meter.avg*100,
    }
    
    return train_info
        

def test(data_loader, model, device, epoch, cfg):
    """
    Test function with BCE loss for multi-label classification
    """
    model.eval()
    
    activities_meter=AverageMeter()
    loss_meter=AverageMeter()
    
    epoch_timer=Timer()
    with torch.no_grad():
        for batch_data in data_loader:
            # prepare batch data
            batch_data=[b.to(device=device) for b in batch_data]
            batch_size=batch_data[0].shape[0]
            
            # Extract labels
            if len(batch_data) == 5:
                images_in, boxes_in, actions_in, activities_in, bboxes_num_in = batch_data[0], batch_data[1], batch_data[2], batch_data[3], batch_data[4]
            else:
                images_in, boxes_in, actions_in, activities_in = batch_data[0], batch_data[1], batch_data[2], batch_data[3]
                bboxes_num_in = None
            
            # Convert to multi-label format
            if len(activities_in.shape) == 2:
                activities_in = activities_in[:, 0]
            
            # Create binary labels
            num_activities = cfg.num_activities
            activities_binary = torch.zeros(batch_size, num_activities, device=device)
            activities_binary.scatter_(1, activities_in.unsqueeze(1), 1)
            
            # Forward
            if bboxes_num_in is not None:
                activities_scores = model((images_in, boxes_in, bboxes_num_in))
            else:
                activities_scores = model((images_in, boxes_in))
            
            # BCE loss
            activities_loss = F.binary_cross_entropy_with_logits(activities_scores, activities_binary)
            
            # Calculate accuracy
            activities_probs = torch.sigmoid(activities_scores)
            activities_pred = (activities_probs > 0.5).float()
            
            correct = (activities_pred == activities_binary).float()
            activities_accuracy = correct.mean(dim=1).mean().item()
            
            activities_meter.update(activities_accuracy, batch_size)
            loss_meter.update(activities_loss.item(), batch_size)

    test_info={
        'time':epoch_timer.timeit(),
        'epoch':epoch,
        'loss':loss_meter.avg,
        'activities_acc':activities_meter.avg*100,
    }
    
    return test_info

