# Labelled PointNet Autoencoder  

This repository provides an implementation of an autoencoder for a labelled point cloud. Unlike usual point cloud autoencoder models which encode the geometric structure of point clouds, this work aims to explore the feasiblity of additionally encoding point label information alongside geometric information.  

Utilising [PointNet](https://arxiv.org/abs/1612.00593) architectures, the model chains a typical point cloud autoencoder with a segmentation model, and trains both simultaneously. The hope is that by combining both architectures, the model learns both geometric and semantic information in tandem.  

## Requirements  

* The appropriate PyTorch version (see instructions from https://pytorch.org/)  
* ```faiss-cpu```  
* ```pandas >= 2.0.0```  
* ```tqdm```  

```train.py``` utilises [PyTorch3D](https://github.com/facebookresearch/pytorch3d) for calculating chamfer distance. If used in any of the contexts below, ensure that PyTorch3D is also installed.  

## Data Format  

Labelled point clouds should be in ```.csv``` format, with each ```.csv``` representing one point cloud, and each row representing one point. All point clouds should be stored in one directory and referenced with the ```--dir``` flag for training.  

| Column | Data type |
|---|---|
| x | ```float``` |
| y | ```float``` |
| z | ```float``` |
| label | ```int``` |

## Use  

### Local development  

For local development or training on CUDA, run the following commands to build a Docker image and run the container:  
```
cd docker
sh docker-build.sh
sh docker-run.sh
```
For CPU development, append ```cpu``` to the end of both ```sh``` commands.  

### HPC training  

For training on HPC, transfer the following files and directories into the target working directory on the HPC system:  
* Data files in an appropriate directory  
* [./datasets/](./datasets/)  
* [./models/](./models/)  
* ```train-vanilla.py``` (or ```train.py``` if the system supports installing PyTorch3D)  

A sample job script ```train.pbs``` is provided.  

### Arguments  

The following describes input parameters for the training scripts.  

| Argument | Default | Flag | Short flag | Type | Description |
|---|---|---|---|---|---|
| Training directory |  | ```--dir``` |  | ```str``` | Relative or absolute path to directory containing training data |
| Number of classes |  | ```--num-classes``` | ```-c``` | ```int``` | Values in the label column should be less than this argument |
| Radius of point cloud | ```100.0``` | ```--radius``` | ```-r``` | ```float``` | x, y, z values are normalised by dividing by this argument |
| Maximum number of points | ```None``` | ```--max-points``` |  | ```int``` | If specified, limits the maximum number of points considered per point cloud via random sampling. If not specified, each point cloud must have the same number of points. |
| Chamfer distance weight | ```0.5``` | ```--cd-weight``` |  | ```float``` | Weight of chamfer distance loss |
| Negative log likelihood weight | ```0.5``` | ```--nll-weight``` |  | ```float``` | Weight of negative log likelihood loss |
| Torch device | ```"cuda"``` | ```--device``` | ```-d``` | ```str``` | Torch device |
| Number of workers | ```1``` | ```--num-workers``` | ```-w``` | ```int``` | Number of subprocesses for data loading |
| Learning rate | ```1E-4``` | ```--lr``` | ```-l``` | ```float``` | Learning rate |
| Batch size | ```50``` | ```--batch-size``` | ```-b``` | ```int``` | Batch size |
| Number of epochs | ```500``` | ```--num-epochs``` | ```-e``` | ```int``` | Number of epochs |
| Name of run | ```"train"``` | ```--name``` |  | ```str``` | Used to name run logs and saved weights |
| Save frequency | ```50``` | ```--save-freq``` |  | ```int``` | Number of epochs per save checkpoint |