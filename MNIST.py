import os
import errno
import argparse
import sys
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split

import numpy as np
from tensorflow.keras.models import load_model

from data_utils import load_MNIST_data, load_EMNIST_data, generate_bal_private_data,\
generate_partial_data
from FedMD import FedMD
from Neural_Networks import train_models, cnn_2layer_fc_model, cnn_3layer_fc_model

DATASET_SIZE = 70000
TRAIN_RATIO = 0.5
VALIDATION_RATIO = 0.4
TEST_RATIO = 0.1

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
X = np.concatenate([x_train, x_test])
y = np.concatenate([y_train, y_test])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=(1-TRAIN_RATIO))
X_ref, X_test, y_ref, y_test = train_test_split(X_val, y_val, test_size=(0.1))

X_train = X_train.astype("float32") / 255.0
X_train = np.reshape(X_train, (-1, 28, 28, 1))

X_test = X_test.astype("float32") / 255.0
X_test = np.reshape(X_test, (-1, 28, 28, 1))

X_ref = X_ref.astype("float32") / 255.0
X_ref = np.reshape(X_ref, (-1, 28, 28, 1))
def parseArg():
    parser = argparse.ArgumentParser(description='FedMD, a federated learning framework. \
    Participants are training collaboratively. ')
    parser.add_argument('-conf', metavar='conf_file', nargs=1, 
                        help='the config file for FedMD.'
                       )

    conf_file = os.path.abspath("conf/EMNIST_balance_conf.json")
    
    if len(sys.argv) > 1:
        args = parser.parse_args(sys.argv[1:])
        if args.conf:
            conf_file = args.conf[0]
    return conf_file

CANDIDATE_MODELS = {"2_layer_CNN": cnn_2layer_fc_model, 
                    "3_layer_CNN": cnn_3layer_fc_model} 

if __name__ == "__main__":
    conf_file =  parseArg()
    with open(conf_file, "r") as f:
        conf_dict = eval(f.read())
        
        #n_classes = conf_dict["n_classes"]
        model_config = conf_dict["models"]
        pre_train_params = conf_dict["pre_train_params"]
        model_saved_dir = conf_dict["model_saved_dir"]
        model_saved_names = conf_dict["model_saved_names"]
        is_early_stopping = conf_dict["early_stopping"]
        public_classes = conf_dict["public_classes"]
        private_classes = conf_dict["private_classes"]
        n_classes = len(public_classes) + len(private_classes)
        
        emnist_data_dir = conf_dict["EMNIST_dir"]    
        N_parties = conf_dict["N_parties"]
        N_samples_per_class = conf_dict["N_samples_per_class"]
        
        N_rounds = conf_dict["N_rounds"]
        N_alignment = conf_dict["N_alignment"]
        N_private_training_round = conf_dict["N_private_training_round"]
        private_training_batchsize = conf_dict["private_training_batchsize"]
        N_logits_matching_round = conf_dict["N_logits_matching_round"]
        logits_matching_batchsize = conf_dict["logits_matching_batchsize"]
        
        
        result_save_dir = conf_dict["result_save_dir"]

    
    del conf_dict, conf_file
    
    public_dataset = {"X": X_ref, "y": y_ref}
    
    #generate private data
    private_data, total_private_data \
    = generate_bal_private_data(X_train, y_train, 
                                N_parties = N_parties,             
                                classes_in_use = private_classes, 
                                N_samples_per_class = N_samples_per_class, 
                                data_overlap = False)
    
    X_tmp, y_tmp = generate_partial_data(X = X_test, y= y_test, 
                                         class_in_use = private_classes, verbose = True)
    private_test_data = {"X": X_tmp, "y": y_tmp}
    del X_tmp, y_tmp
    
    parties = []
    if model_saved_dir is None:
        for i, item in enumerate(model_config):
            model_name = item["model_type"]
            model_params = item["params"]
            tmp = CANDIDATE_MODELS[model_name](n_classes=n_classes, 
                                               input_shape=(28,28),
                                               **model_params)
            print("model {0} : {1}".format(i, model_saved_names[i]))
            print(tmp.summary())
            parties.append(tmp)
            
            del model_name, model_params, tmp
        #END FOR LOOP
        pre_train_result = train_models(parties, 
                                        X_train, y_train, 
                                        X_test, y_test,
                                        save_dir = model_saved_dir, save_names = model_saved_names,
                                        early_stopping = is_early_stopping,
                                        **pre_train_params
                                       )
    else:
        dpath = os.path.abspath(model_saved_dir)
        model_names = os.listdir(dpath)
        for name in model_names:
            tmp = None
            tmp = load_model(os.path.join(dpath ,name))
            parties.append(tmp)
    
    #del  X_train_MNIST, y_train_MNIST, X_test_MNIST, y_test_MNIST, \
    #X_train_EMNIST, y_train_EMNIST, X_test_EMNIST, y_test_EMNIST, writer_ids_train, writer_ids_test
    
    
    fedmd = FedMD(parties, 
                  public_dataset = public_dataset,
                  private_data = private_data, 
                  total_private_data = total_private_data,
                  private_test_data = private_test_data,
                  N_rounds = N_rounds,
                  N_alignment = N_alignment, 
                  N_logits_matching_round = N_logits_matching_round,
                  logits_matching_batchsize = logits_matching_batchsize, 
                  N_private_training_round = N_private_training_round, 
                  private_training_batchsize = private_training_batchsize)
    
    initialization_result = fedmd.init_result
    pooled_train_result = fedmd.pooled_train_result
    
    collaboration_performance = fedmd.collaborative_training()
    
    if result_save_dir is not None:
        save_dir_path = os.path.abspath(result_save_dir)
        #make dir
        try:
            os.makedirs(save_dir_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise    
    
        