## Usage

To run experiments, you will likely need to change the following keys in `config.json`:

+ num_loader_workers: determines number of processes to use for data loading. Set it to the number of cpus on your instance machine.
+ path: directory where output will be stored
+ retinal_path: path to directory containing the images
+ class_column: name of the column containing the class labels in metadata file. Will need to change this depending on whether classifier is for disease or gender.
+ n_classes: number of classes.
+ class_dist: need to adjust this depending on whether doing disease or gender experiment.