# SpeechCommand-recognition
Different DL models for speech command dataset. provide training, testing and live inference scripts.

## Download Dataset
To download the dataset go on the root directory and execute :

    mkdir data
    cd data
    wget http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz
    tar -xvf <name_untar_dir>
    
    
## Running supervised training 
### Configuration

    In the config file under the supervised_learning directory you can set the different hyperparameters for the training such as ;
        - BATCH SIZE
        - NUMBER OF EPOCHS
        ...
        
### Runing the training
During the first training the dataset is serialize under the indir model to speed up futur training
After parametring the training you can start the training with 
        
        - python train.py --indir <datase_path>  --serialize <True|False> \
         --checkpoint_path <checkpoint_file> or --model <model_path> --model_type <cnn | lstm | attention_lstm}>
         
You have different models available in supervised_learnings/model you can used a different one changing the train.py