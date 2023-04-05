#!/usr/bin/env python

# import relevant packages
from utils import *
from model import *
# define functions and classes only relevant to training
from tqdm import tqdm
import torchmetrics
import wandb
from autothreshold_f1 import AutoThresholdF1
from torchmetrics import AUROC, AveragePrecision
from typing import Dict, Tuple
from attrdict import AttrDict
def main(config):
    """Evals a model using a config file
    Parameters
    ----------
    config_path : str, optional
        The path to the config file (default is train.conf)
    Returns
    -------
    thing_to_return
        description of thing to return
    """
    #Set Seed
    seed_everything(config['seed'])
    #DEVICE & WanB runs set up
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    wandb.init(project="Catheter-malpositioning-detection", entity="mit_final_project")
    wandb.config = {
        "model":config['model'],
        'k' :config['k']
    }
    wandb.run.name=f"model {wandb.config['model']} eval"
    #Read train val test data
    df_all, df_train_val = readData(config['path'])
    _, df_test= readTestData(config['path'])    
    val_perc = None if config['val_perc'] == "None" else config['val_perc']
    folds = k_fold_cross_val(df_train_val, df_all, k=config['k'], stratified_grouped=config['stratified_grouped'], val_perc=val_perc)
    #defining our metrics to evaluate on 
    metrics = AttrDict({
      "auroc": AUROC(num_classes=11, average="macro", compute_on_step=True),
      "f1": AutoThresholdF1(num_classes=11, average="macro", compute_on_step=True),
      "auprc": AveragePrecision(num_classes=11, average="macro", compute_on_step=True)
    })
    test_loader=loadTestData(df_test, df_all,  config['batch_size'], config['image_size'])
    for curr_fold in range(len(folds)):
        print('Evaluating on Fold ' + str(curr_fold + 1) + ' of ' + str(len(folds)))
        _, valid_loader = loadData(
            folds[curr_fold], df_all, config['batch_size'], config['image_size'])
        
            # Model
        if config['model'] == 'resnet50':
            model = resnet50()
        elif config['model'] == 'resnet50_binary':
            model = resnet50_binary()
        elif config['model'] == 'resnet200d':
            model = ResNet200D()
        elif config['model'] == 'densenet121':
            model = densenet121()
        elif config['model'] == 'EfficientNetB5':
            model = EfficientNetB5()
        elif config['model'] == 'gloria':
            model = gloria_model()
        else:
            raise NotImplementedError
       
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        PATH=config["saved_path"]
        K_PATH=config['model']+"k_" + str(curr_fold) + "_path"
        MODEL_PATH=str(PATH+K_PATH)
        #  print(torch.load(MODEL_PATH).items())
        state = torch.load(MODEL_PATH)
        model.load_state_dict(state['model'],strict=False)
        model.to(device)
        model.eval()
        print('Start validating on held out validation set...')
        lp=[]
        ll=[]
        with torch.no_grad():
            for i, batch in enumerate(tqdm(valid_loader)):
                X,y= batch
                X = X.float().to(device)
                y = y.long().to(device)
                preds = model(X).reshape((X.size(0), 11,2))
                y_probs = preds.softmax(dim=1).float()
                lp.append(y_probs)
                ll.append(y.long())
        y_probs=torch.cat(lp, dim=0)
        y_long=torch.cat(ll, dim=0)
        for i, (metric_name, metric) in enumerate(metrics.items()):
            m = metric(y_probs[:,:,1],y_long)
            wandb.log({"val fold"+metric_name : m})
            print(m)
            metric.reset()
        print('Start validating on test set...')
        lp=[]
        ll=[]
        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_loader)):
                X,y= batch
                X = X.float().to(device)
                y = y.long().to(device)
                preds = model(X).reshape((X.size(0), 11,2))
                y_probs = preds.softmax(dim=1).float()
                lp.append(y_probs)
                ll.append(y.long())
        y_probs=torch.cat(lp, dim=0)
        y_long=torch.cat(ll, dim=0)
        for i, (metric_name, metric) in enumerate(metrics.items()):
            m = metric(y_probs[:,:,1],y_long)
            wandb.log({"test fold"+metric_name : m})
            print(m)
            metric.reset()


if __name__ == "__main__":
    # train k networks reading from a config file for parameters
    config = read_json('./eval.json')
    main(config)
