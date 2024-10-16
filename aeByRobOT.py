import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import copy
import time
import random
import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
from mixup_utils import compute_loss, compute_loss_100, compute_acc

np.set_printoptions(threshold=np.inf)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


if __name__ == '__main__':
    param_dict = {"dataset_name": ["fashion_mnist", "svhn", "cifar10", "cifar100"],
                  "model_architecture": ["mobilenetv2", "shufflenetv2", "resnet20", "resnet56"],
                  "num_classes": [10, 10, 10, 100],
                  "ep": [0.1, 0.1, 0.1, 0.1],
                  "lr": [0.01, 0.01, 0.01, 0.01]}

    for da_type in ["mixup", "cutmix"]:
        for param_index in [0, 1, 2, 3]:
            dataset_name = param_dict["dataset_name"][param_index]
            model_architecture = param_dict["model_architecture"][param_index]
            num_classes = param_dict["num_classes"][param_index]
            lr = param_dict["lr"][param_index]
            ep = param_dict["ep"][param_index]
            fuzzer_name = "robot"

            # path
            data_architecture = dataset_name + "_" + model_architecture
            base_path = "./checkpoint/" + data_architecture
            original_model_path = base_path + "/" + da_type + "/models/" + data_architecture + ".h5"
            generated_seed_path = base_path + "/" + da_type + "/generated_seed/"
            ae_data_path = base_path + "/" + da_type + "/" + fuzzer_name + "/"
            if not os.path.exists(ae_data_path):
                os.makedirs(ae_data_path)
            print(f"Generate by {fuzzer_name}, Model architecture: {model_architecture}, dataset: {dataset_name}, num_classes: {num_classes}, ep: {ep}, lr: {lr}")

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
                model = tf.keras.models.load_model(original_model_path, custom_objects={'compute_loss_100': compute_loss_100, 'compute_acc': compute_acc})
            else:
                model = tf.keras.models.load_model(original_model_path, custom_objects={'compute_loss': compute_loss, 'compute_acc': compute_acc})

            batch_size = 1000
            data_type_list = ["train", "dlfuzz_seeds", "adapt_seeds", "robot_seeds", "basil_seeds"] 
            total_duration = 3000  # fuzzing durations
            num = 4000  # max number of seeds
            for data_type in data_type_list:
                print(f"Fuzzer: {fuzzer_name}, Data type: {data_type}, Model architecture: {model_architecture}, dataset: {dataset_name}, num_classes: {num_classes}, ep: {ep}")
                total_sets = []
                random.seed(2023)
                all_indexes = np.array(random.sample(list(range(len(x_train))), num))

                if data_type == "train":
                    original_seed = x_train[all_indexes]
                    original_seed_gt = y_train[all_indexes]
                    replacement_seed = x_train[all_indexes]
                else:
                    with np.load(generated_seed_path + "/" + data_type + ".npz", allow_pickle=True) as f:
                        generated_seeds, seed_ground_truth, seed_original_pred_label, seed_errors = f['generated_seeds'], f['ground_truth'], f['original_pred_label'], f['errors']

                    if data_type != "basil_seeds":
                        replacement_seed = generated_seeds
                    else:
                        replacement_seed = generated_seeds[all_indexes]

                time_start = time.time()
                processed_seed = 0
                for i in range(int(np.ceil(len(replacement_seed)/batch_size))):
                    seeds = np.array(range(replacement_seed.shape[0]))[i*batch_size: (i+1)*batch_size]
                    original_images = original_seed[seeds]
                    replacement_images = replacement_seed[seeds]
                    labels = original_seed_gt[seeds]

                    # some training samples is static, i.e., grad=<0>, hard to generate.
                    gen_img = tf.Variable(replacement_images)
                    original_prediction_labels_one_hot = tf.keras.utils.to_categorical(np.argmax(model(replacement_images), axis=1), num_classes)
                    with tf.GradientTape() as g:
                        loss = tf.keras.losses.categorical_crossentropy(original_prediction_labels_one_hot, model(gen_img))
                        grads = g.gradient(loss, gen_img)

                    fols = np.linalg.norm((grads.numpy()+1e-20).reshape(replacement_images.shape[0], -1), ord=2, axis=1)
                    seeds_filter = np.where(fols > 1e-3)[0]

                    lam = 1
                    top_k = 5
                    steps = 3
                    for idx in seeds_filter:
                        img_list = []
                        orig_img = copy.deepcopy(original_images[[idx]])
                        orig_index = np.argmax(model(orig_img)[0])
                        orig_norm = np.linalg.norm(orig_img)
                        tmp_img = replacement_images[[idx]]
                        img_list.append(tf.identity(tmp_img))
                        logits = model(tmp_img)
                        tmp_img_index = np.argmax(logits[0])
                        target = tf.keras.utils.to_categorical([tmp_img_index], num_classes)
                        label_top5 = np.argsort(logits[0])[-top_k:-1][::-1]
                        folMAX = 0.0
                        processed_seed += 1

                        while len(img_list) > 0:
                            gen_img = img_list.pop(0)

                            for _ in range(steps):
                                gen_img = tf.Variable(gen_img, dtype=float)
                                with tf.GradientTape(persistent=True) as g:
                                    loss = tf.keras.losses.categorical_crossentropy(target, model(gen_img))
                                    grads = g.gradient(loss, gen_img)
                                    fol = tf.norm(grads+1e-20)
                                    g.watch(fol)
                                    logits = model(gen_img)
                                    obj = lam*fol - logits[0][tmp_img_index]
                                    dl_di = g.gradient(obj, gen_img)
                                del g

                                gen_img = gen_img + dl_di * lr * (random.random() + 0.5)
                                gen_img = tf.clip_by_value(gen_img, clip_value_min=0.0, clip_value_max=1.0)

                                with tf.GradientTape() as t:
                                    t.watch(gen_img)
                                    loss = tf.keras.losses.categorical_crossentropy(target, model(gen_img))
                                    grad = t.gradient(loss, gen_img)
                                    fol = np.linalg.norm(grad.numpy())  # L2 adaption

                                distance = np.linalg.norm(gen_img.numpy() - orig_img) / orig_norm
                                if fol > folMAX and distance < ep:
                                    folMAX = fol
                                    img_list.append(tf.identity(gen_img))

                                if distance < ep:
                                    gen_index = np.argmax(model(gen_img)[0])
                                    if gen_index != orig_index:
                                        total_sets.append((idx + batch_size * i, time.time() - time_start, fol, gen_img.numpy(), labels[idx], gen_index, processed_seed))
                        print(f"---------------------Time: {time.time()-time_start}, {idx + i*batch_size}, {len(total_sets)}----------------------")
                        if time.time() - time_start >= total_duration:
                            break
                    print(f"Current length of total_sets: {len(total_sets)}")
                    idx_all = np.array([item[0] for item in total_sets])
                    time_all = np.array([item[1] for item in total_sets])
                    fol_all = np.array([item[2] for item in total_sets])
                    ae_all = np.array([item[3][0] for item in total_sets])
                    label_all = np.array([item[4] for item in total_sets])
                    pred_label_all = np.array([item[5] for item in total_sets])
                    processed_seed_all = np.array([item[6] for item in total_sets])
                    np.savez(ae_data_path + data_type + "_ae.npz", idx=idx_all, time=time_all, ae=ae_all, ae_label=label_all, pred_label=pred_label_all, processed_seed=processed_seed_all, fol=fol_all)

                    for j in range(0, total_duration, 1000):
                        cur_idxs = np.where(np.array(time_all) <= (j + 1000))[0]
                        ae_train_idxs = idx_all[cur_idxs]
                        ae_pred_labels = pred_label_all[cur_idxs]
                        processed_seeds = processed_seed_all[cur_idxs]
                        all_labels = [len(set(ae_pred_labels[np.where(ae_train_idxs == one_ae_train_idx)[0]])) for one_ae_train_idx in list(set(ae_train_idxs))]
                        print(f"Time: {total_duration}, seed num: {max(processed_seeds)}, ae num: {len(ae_train_idxs)}, "
                              f"Unsuccessful rate: {round((1 - len(set(ae_train_idxs)) / max(processed_seeds)) * 100, 2)}, Category: {len(set(ae_train_idxs))}, "
                              f"Label: {np.sum(all_labels)}, unsuccessful: {max(processed_seeds)-len(set(ae_train_idxs))}")

                    if time.time() - time_start >= total_duration:
                        break
