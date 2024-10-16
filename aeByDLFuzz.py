import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import time
import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from utils.adapt import Network
from utils.adapt.metric import NC, TKNC
from utils.adapt.fuzzer import WhiteBoxFuzzer
from utils.adapt.strategy import AdaptiveParameterizedStrategy, UncoveredRandomStrategy, MostCoveredStrategy
from mixup_utils import compute_loss, compute_loss_100, compute_acc

np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


if __name__ == '__main__':
    param_dict = {"dataset_name": ["fashion_mnist", "svhn", "cifar10", "cifar100"],
                  "model_architecture": ["mobilenetv2", "shufflenetv2", "resnet20", "resnet56"],
                  "num_classes": [10, 10, 10, 100],
                  "ep": [0.1, 0.1, 0.1, 0.1],
                  "lr": [0.01, 0.01, 0.01, 0.01],
                  "duration": [5, 5, 10, 10]}

    for da_type in ["mixup", "cutmix"]:     # "mixup", "cutmix"
        for param_index in [0, 1, 2, 3]:
            dataset_name = param_dict["dataset_name"][param_index]
            model_architecture = param_dict["model_architecture"][param_index]
            num_classes = param_dict["num_classes"][param_index]
            ep = param_dict["ep"][param_index]
            lr = param_dict["lr"][param_index]
            duration = param_dict["duration"][param_index]
            fuzzer_name = "dlfuzz"

            # path
            data_architecture = dataset_name + "_" + model_architecture
            base_path = "./checkpoint/" + data_architecture
            original_model_path = base_path + "/" + da_type + "/models/" + data_architecture + ".h5"
            generated_seed_path = base_path + "/" + da_type + "/generated_seed/"
            ae_data_path = base_path + "/" + da_type + "/" + fuzzer_name + "/"
            if not os.path.exists(ae_data_path):
                os.makedirs(ae_data_path)

            original_data_path = base_path + "/dataset/" + dataset_name + ".npz"
            with np.load(original_data_path) as f:
                x_train, y_train, x_test, y_test = f['x_train'], f['y_train'], f['x_test'], f['y_test']

            x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=2023)
            x_train = x_train.astype('float32') / 255
            x_valid = x_valid.astype('float32') / 255
            x_test = x_test.astype('float32') / 255

            if dataset_name == "fashion_mnist":
                x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
                x_valid = x_valid.reshape(x_valid.shape[0], x_valid.shape[1], x_valid.shape[2], 1)
                x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

            y_train = tf.keras.utils.to_categorical(y_train, num_classes)
            y_valid = tf.keras.utils.to_categorical(y_valid, num_classes)
            y_test = tf.keras.utils.to_categorical(y_test, num_classes)

            if param_index == 3:
                network = Network(tf.keras.models.load_model(original_model_path, custom_objects={'compute_loss_100': compute_loss_100, 'compute_acc': compute_acc}))
            else:
                network = Network(tf.keras.models.load_model(original_model_path, custom_objects={'compute_loss': compute_loss, 'compute_acc': compute_acc}))

            # generate adversarial examples at once.
            data_type_list = ["train", "dlfuzz_seeds", "adapt_seeds", "robot_seeds", "basil_seeds"] 
            total_duration = 3000  # fuzzing durations
            num = 4000  # max number of seeds
            for data_type in data_type_list:
                print(f"Fuzzer: {fuzzer_name}, Data type: {data_type}, Model architecture: {model_architecture}, dataset: {dataset_name}, num_classes: {num_classes}, ep: {ep}")
                random.seed(2023)
                all_indexes = np.array(random.sample(list(range(len(x_train))), num))
                original_seed = x_train[all_indexes]
                original_seed_gt = y_train[all_indexes]

                if data_type == "train":
                    replacement_seed = x_train[all_indexes]
                else:
                    with np.load(generated_seed_path + "/" + data_type + ".npz", allow_pickle=True) as f:
                        generated_seeds, seed_ground_truth, seed_original_pred_label, seed_errors = f['generated_seeds'], f['ground_truth'], f['original_pred_label'], f['errors']

                    if data_type != "basil_seeds":
                        replacement_seed = generated_seeds
                    else:
                        replacement_seed = generated_seeds[all_indexes]

                idx_all = []
                ae_all = []
                ae_label_all = []
                ae_pred_label_all = []
                ae_time_all = []
                ground_truth_a_all = []
                ground_truth_b_all = []
                lam_all = []
                mixed_indexes_all = []

                metric = NC(0.5)
                # metric = TKNC()

                time_start = time.time()
                for i in range(len(original_seed)):
                    if fuzzer_name == "adapt":
                        strategy = AdaptiveParameterizedStrategy(network)
                    elif fuzzer_name == "deepxplore":
                        strategy = UncoveredRandomStrategy(network)
                    elif fuzzer_name == "dlfuzz":
                        strategy = MostCoveredStrategy(network)
                    else:
                        strategy = None

                    fuzzer = WhiteBoxFuzzer(network=network, original_input=original_seed[i], replacement_input=replacement_seed[i], ground_truth=original_seed_gt[i], metric=metric, strategy=strategy, k=10, delta=ep, class_weight=0.5, neuron_weight=0.5, lr=lr, trail=3, decode=None)
                    ae, ae_label, _, ae_pred_label = fuzzer.start(seconds=duration, append='min_dist')

                    idx_all.extend(np.array([i]*len(ae)))
                    ae_all.extend(ae)
                    ae_label_all.extend(ae_label)
                    ae_pred_label_all.extend(ae_pred_label)
                    ae_time_all.extend(np.array([time.time() - time_start] * len(ae)))
                    print(f"---------------------{i}, {len(ae)}, {len(ae_all)}----------------------")
                    if time.time() - time_start >= total_duration:
                        break

                print(f"length of AE {len(ae_all)}")
                np.savez(ae_data_path + data_type + "_ae.npz", idx=np.array(idx_all), ae=np.array(ae_all), ae_label=np.array(ae_label_all), time=np.array(ae_time_all), pred_label=np.array(ae_pred_label_all))

                for j in range(0, total_duration, 1000):
                    cur_idxs = np.where(np.array(ae_time_all) <= (j + 1000))[0]
                    ae_train_idxs = np.array(idx_all)[cur_idxs]
                    ae_pred_labels = np.array(ae_pred_label_all)[cur_idxs]
                    all_labels = [len(set(ae_pred_labels[np.where(ae_train_idxs == one_ae_train_idx)[0]])) for one_ae_train_idx in list(set(ae_train_idxs))]
                    print(f"Time: {j + 1000}, seed num: {max(ae_train_idxs) + 1}, ae num: {len(ae_train_idxs)},"
                          f"Unsuccessful rate: {round((1-len(set(ae_train_idxs))/max(ae_train_idxs))*100, 2)}, Category: {len(set(ae_train_idxs))}, "
                          f"Label: {np.sum(all_labels)}, unsuccessful: {max(ae_train_idxs)+1-len(set(ae_train_idxs))}")

