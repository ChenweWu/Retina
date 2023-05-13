#!/usr/bin/env python

# import relevant packages
from utils import *
from model import *
# define functions and classes only relevant to training
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


def main(config):
    """Evaluate a model using a config file."""
    # DEVICE
    DEVICE = config['device']
    base_path = config['retinal_path']
    test_folder = config['test_folder']

    image_path = f'{base_path}{test_folder}/'

    df_val = readTestData(config['test_path'])
    valid_loader = loadRetinalData2(df_val, 1, config['image_size'], image_path, config['class_column'],
                                  config['channel_avg'], config['channel_std'], config['crop_dims'], split='val',
                                  num_workers=config['num_loader_workers'])

    # Load our trained model...
    if config['model'] == 'resnet50':
        model = resnet50()
    elif config['model'] == 'resnet50_binary':
        model = resnet50_binary()
    elif config['model'] == 'resnet200d':
        model = ResNet200D(config['n_classes'])
    elif config['model'] == 'densenet121':
        model = densenet121()
    elif config['model'] == 'EfficientNetB5':
        model = EfficientNetB5()
    elif config['model'] == 'inceptionv3':
        model = inceptionv3()
    else:
        raise NotImplementedError

    MODEL_PATH = config['path'] + config['my_model']
    print("Load model from: " + MODEL_PATH)
    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint['model'],strict=True)
    model = model.to(DEVICE)

    for epoch in range(1):
        model.eval()
        with torch.no_grad():
            print(f"Start validating... ")
            y_list = []
            pred_list = []
            for X, y in tqdm(valid_loader,total=len(valid_loader)):
                X = X.to(DEVICE)
                y = y.to(DEVICE)

                pred = model(X)
                softmax = torch.nn.Softmax(dim=1)

                # TODO: if has error here, try softmax(pred.data)
                softmax_pred= softmax(pred).cpu().detach().numpy()
                num_pred = np.argmax(softmax_pred,axis=1).tolist()
                origs = y.cpu().detach().numpy().tolist()
                y_list.extend(origs)
                pred_list.extend(num_pred)
            print("EPOCH: " + str(epoch))
            print("Confusion Matrix: ", confusion_matrix(y_list, pred_list))
            print("Accuracy: ", accuracy_score(y_list, pred_list))
            output_file_name = f'{test_folder}_evaluation.txt'
            with open(config['path'] + output_file_name, 'a') as f:
                f.write("")
                f.write(np.array2string(confusion_matrix(y_list, pred_list), separator=', '))
                f.write(f'Accuracy: {accuracy_score(y_list, pred_list)}')
if __name__ == "__main__":
    # train k networks reading from a config file for parameters
    config = read_json('./eval_config.json')
    main(config)
