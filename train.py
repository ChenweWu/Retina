#!/usr/bin/env python

# import relevant packages
from utils import *
from model import *
# define functions and classes only relevant to training
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score
def main(config):
    """Trains a model using a config file

    Parameters
    ----------
    config_path : str, optional
        The path to the config file (default is train.conf)

    Returns
    -------
    thing_to_return
        description of thing to return
    """
    # DEVICE
    DEVICE = torch.device("cuda:1")

     # Seed
    seed_everything(config['seed'])
    
    # Load data
    df_all = readData(config['train_path'])
    df_val = readTestData(config['test_path'])
    # val_perc = None if config['val_perc']=="None" else config['val_perc']
    # folds = k_fold_cross_val(df_train_val, df_all, k=config['k'], stratified_grouped = config['stratified_grouped'], val_perc=val_perc)
    
    # for curr_fold in range(len(folds)):
    #     print('Training on Fold ' + str(curr_fold + 1) + ' of ' + str(len(folds)))
    train_loader = loadRetinalData2( df_all, config['batch_size'], config['image_size'], config['retinal_path'], config['channel_avg'], config['channel_std'], split='train')
    val_loader = loadRetinalData2( df_val, 1, config['image_size'], config['retinal_path'], config['channel_avg'], config['channel_std'], split='val')
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
    elif config['model'] == 'inceptionv3':
        model = inceptionv3()
    else:
        raise NotImplementedError
    
    
    model = model.to(DEVICE)
    # Loss Fc
    class_dis = np.array(config['class_dist'])
    class_weights =1-class_dis/np.sum(class_dis)
    print(class_weights)
    criterion = get_lossfn(config['loss'],torch.tensor(class_weights).float().to(DEVICE))

    # Get optim
    optimizer = get_optim(model, config['optimizer'], config['lr'])

    # Train
    best_loss = np.inf
    best_f1 = 0
    valid_loss = 0
    NUM_EPOCH = config['NUM_EPOCH']
    for epoch in range(NUM_EPOCH):
        model.train()
        print(f"Start training EPOCH {epoch}...")
        y_list = []
        pred_list = []
        for X, y in tqdm(train_loader,total=len(train_loader)):
            optimizer.zero_grad()
            X = X.float().to(DEVICE)
            y = y.long().to(DEVICE)
#                 pred = model(X).reshape((X.size(0), 11,2))
#                 loss = 0
#                 for i in range(pred.size(1)):
#                     loss += criterion(pred[:,i], y[:,i])
            pred = model(X)
         #   print(pred.shape)
            softmax = torch.nn.Softmax(dim=1)
            softmax_pred= softmax(pred).cpu().detach().numpy()
         #   print(softmax_pred)
            num_pred = np.argmax(softmax_pred,axis=1).tolist()
            origs = y.cpu().detach().numpy().tolist()
            y_list.extend(origs)
        #    print(y_list)
            pred_list.extend(num_pred)
           # print(num_pred)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
           # valid_loss += loss.item()
        valid_loss /= (len(train_loader)*config['batch_size'])
        print(f"EPOCH:{epoch}, Loss:{valid_loss}")
        print(confusion_matrix(y_list, pred_list))
        print(accuracy_score(y_list, pred_list))
        model.eval()
        with torch.no_grad():
            print(f"Start validating epoch{epoch}")
            y_list = []
            pred_list = []
            for X, y in tqdm(val_loader,total=len(val_loader)):
                optimizer.zero_grad()
                X = X.float().to(DEVICE)
                y = y.long().to(DEVICE)
    #                 pred = model(X).reshape((X.size(0), 11,2))
    #                 loss = 0
    #                 for i in range(pred.size(1)):
    #                     loss += criterion(pred[:,i], y[:,i])
                pred = model(X)
            #   print(pred.shape)
                softmax = torch.nn.Softmax(dim=1)
                softmax_pred= softmax(pred).cpu().detach().numpy()
            #   print(softmax_pred)
                num_pred = np.argmax(softmax_pred,axis=1).tolist()
                origs = y.cpu().detach().numpy().tolist()
                y_list.extend(origs)
            #    print(y_list)
                pred_list.extend(num_pred)
            # print(num_pred)
                # loss = criterion(pred, y)
                # loss.backward()
                optimizer.step()
            #  valid_loss += loss.item()
        #   valid_loss /= (len(train_loader)*config['batch_size'])
        # print(f", Loss:{valid_loss}")
        print(confusion_matrix(y_list, pred_list))
        print(accuracy_score(y_list, pred_list))
        f1 = f1_score(y_list, pred_list, average='macro')
        print("f1 score: ", f1)
        print("f1 score: ", f1)

        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, config['path'] +config['model']+' epoch '+str(epoch)+"f1 {}".format(f1)+"full wd")
        # torch.save(model.state_dict(), config['model']+"k_" + str(curr_fold) + ".pth")
        print("saved...")
        with open("/home/ubuntu/e3.txt", 'a') as f:
            f.write("")
            f.write(f"epoch {epoch}, f1 score: {f1}")
            f.write(np.array2string(confusion_matrix(y_list, pred_list), separator=', '))
        # model.eval()
        # valid_loss = 0
#             print(f"Start validating on validation set...")
#             with torch.no_grad():
#                 for X, y in valid_loader:
#                     X = X.float().to(DEVICE)
#                     y = y.long().to(DEVICE)
# #                     loss = 0
# #                     pred = model(X).reshape((X.size(0), 11,2))
# #                     for i in range(pred.size(1)):
# #                         loss += criterion(pred[:,i], y[:,i])
#                     pred = model(X)
#                     loss = criterion(pred, y)                    
#                     valid_loss += loss.item()
#             valid_loss /= len(valid_loader)
#             print(f"EPOCH:{epoch}, Loss:{valid_loss}")
#             if valid_loss < best_loss:
#                 best_loss = valid_loss
#                 state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
#                 torch.save(state, config['model']+"k_" + str(curr_fold) + "_path")
#                 # torch.save(model.state_dict(), config['model']+"k_" + str(curr_fold) + ".pth")
#                 print("saved...")

if __name__ == "__main__":
    # train k networks reading from a config file for parameters
    config = read_json('./config.json')
    main(config)
