#coding=utf-8
import tensorflow as tf
from data_util import *
from model_v4 import model_ig
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import time
import datetime
import numpy as np
import pandas as pd
import random
import json
import pdb
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.set_printoptions(threshold='nan')


flags = tf.app.flags
flags.DEFINE_float("learning_rate", 0.005, "Learning rate for Adam Optimizer.")
flags.DEFINE_float("max_grad_norm", 10.0, "Clip gradients to this norm.")
flags.DEFINE_integer("batch_size", 8, "Batch size for training.")
flags.DEFINE_integer("epochs", 7, "Number of epochs to train for.")
flags.DEFINE_integer("embedding_size", 300, "Embedding size for embedding matrices.")
flags.DEFINE_integer("hidden_dim", 350, "Dimension of hidden state in lstm.")
flags.DEFINE_string("checkpoint_dir", "checkpoints", "checkpoint directory [checkpoints]")
flags.DEFINE_integer("evaluate_every", 500, "Evaluate and print results every x epochs")
flags.DEFINE_integer("checkpoint_every", 500, "Save model after this many steps (default: 100)")
flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
flags.DEFINE_integer("pretrain_embedding", 0, "Load pretrain embedding or not.")
flags.DEFINE_integer("vocab_size", 17199, "Load pretrain embedding or not.")
flags.DEFINE_integer("update_embedding", False, "Train embedding or not.")
flags.DEFINE_integer("num_tags", 17, "2-classification or 17-classification.")
flags.DEFINE_integer("l2_lambda", 1e-4, "Regularization coefficient.")

FLAGS = flags.FLAGS

# 载入数据并区分训练、验证、测试集
data = pd.read_csv('./seq_data_tjz_fill_p_5_2.csv', encoding="utf_8_sig")

xml_number = data.xml_number
ID = data.ID
utt_index = data.utt_index
label_g = data.label_g
label_i = data.label_i
memory = data.memory
x_part = data.x_part
memory_part = data.memory_part

print('***:', len(xml_number))
xml_number = [xml_number[i] for i in range(len(xml_number))]
utt_index = [list(map(eval, utt_index[i].lstrip('[').rstrip(']').split(', '))) for i in range(len(utt_index))]
utt_p = [list(map(eval, x_part[i].lstrip('[').rstrip(']').split(', '))) for i in range(len(x_part))]
label_g = [list(map(eval, label_g[i].lstrip('[').rstrip(']').split(', '))) for i in range(len(label_g))]
label_i = [list(map(eval, label_i[j].lstrip('[').rstrip(']').split(', '))) for j in range(len(label_i))]
print('label_i:', len(label_i))
memory_g = []
memory_g_p = []
for j in range(0, len(memory)):
    each_memory = []
    each_memory_p = []
    memory_list = memory[j].split('], [')
    memory_p_list = memory_part[j].split('], [')
    for k in range(0, len(memory_list)):
        each_memory.append(list(map(eval, memory_list[k].lstrip('[').rstrip(']').split(', '))))
        each_memory_p.append(list(map(eval, memory_p_list[k].lstrip('[').rstrip(']').split(', '))))
    memory_g.append(each_memory)
    memory_g_p.append(each_memory_p)

train_data = []
dev_data = []
test_data = []

train_labelg = []
dev_labelg = []
test_labelg = []

train_labeli = []
dev_labeli = []
test_labeli = []

train_memoryg = []
dev_memoryg = []
test_memoryg = []

train_data_part = []
dev_data_part = []
test_data_part = []

train_memoryg_part = []
dev_memoryg_part = []
test_memoryg_part = []
loss_train = []
loss_dev = []
for j, num in enumerate(xml_number):
    if num>=99 and num < 586:
        train_data.append(utt_index[j])
        train_labelg.append(label_g[j])
        train_labeli.append(label_i[j])
        train_memoryg.append(memory_g[j])
        train_data_part.append(utt_p[j])
        train_memoryg_part.append(memory_g_p[j])
    elif num >= 586:
        dev_data.append(utt_index[j])
        dev_labelg.append(label_g[j])
        dev_labeli.append(label_i[j])
        dev_memoryg.append(memory_g[j])
        dev_data_part.append(utt_p[j])
        dev_memoryg_part.append(memory_g_p[j])
    else:
        test_data.append(utt_index[j])
        test_labelg.append(label_g[j])
        test_labeli.append(label_i[j])
        test_memoryg.append(memory_g[j])
        test_data_part.append(utt_p[j])
        test_memoryg_part.append(memory_g_p[j])
print('data load finished')
print('train_size: ', len(train_data))
print('dev_size: ', len(dev_data))
print('test_size: ', len(test_data))

with open("vocab_fill_p.json", 'r') as f:
    str1 = f.read()
    my_dict = json.loads(str1)
dict_word = []
dict_index = []
for a,b in my_dict.items():
    dict_word.append(a)
    dict_index.append(b)

if FLAGS.pretrain_embedding == 'random':
    embeddings = random_embedding(FLAGS.vocab_size, FLAGS.embedding_size)
else:
   embedding_path = 'np.npy'
   embeddings = np.array(np.load(embedding_path), dtype='float32')

with tf.Session() as sess:
    model_ig = model_ig(FLAGS.batch_size, embeddings, FLAGS.embedding_size, FLAGS.update_embedding, FLAGS.hidden_dim,
                  FLAGS.num_tags, FLAGS.l2_lambda)

    # Define Training procedure
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(0.0003)
    grads_and_vars = optimizer.compute_gradients(model_ig.loss)
    clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], FLAGS.max_grad_norm), gv[1]) for gv in grads_and_vars]
    train_op = optimizer.apply_gradients(clipped_grads_and_vars, global_step=global_step)

    # Keep track of gradient values and sparsity (optional)
    #grad_summaries = []
    #for g, v in clipped_grads_and_vars:
    #    if g is not None:
    #        grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
    #        sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
    #        grad_summaries.append(grad_hist_summary)
    #        grad_summaries.append(sparsity_summary)
    #grad_summaries_merged = tf.summary.merge(grad_summaries)

    # Output directory for models and summaries
    #timestamp = str(int(time.time()))
    #out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    #print("Writing to {}\n".format(out_dir))

    # Summaries for loss and accuracy
    #loss_summary = tf.summary.scalar("loss", model_ig.loss)
    #acc_summary = tf.summary.scalar("accuracy", model_ig.accuracy)

    # Train Summaries
    #train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
    #train_summary_dir = os.path.join(out_dir, "summaries", "train")
    #train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    # Dev summaries
    #dev_summary_op = tf.summary.merge([loss_summary])
    #dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    #dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    #checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    #checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    #if not os.path.exists(checkpoint_dir):
    #    os.makedirs(checkpoint_dir)
    #saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    def train_step(t_x_batch, t_x_batch_part, t_y_batch, t_y_batch_i, t_memory, t_memory_part, t_seq_length, t_m_seq_len):
        feed_dict = {
            model_ig.input_x: t_x_batch,
            model_ig.input_x_part: t_x_batch_part,
            model_ig.input_yg: t_y_batch,
            model_ig.input_yi: t_y_batch_i,
            model_ig.memory: t_memory,
            model_ig.input_memory_part: t_memory_part,
            model_ig.sequence_lengths: t_seq_length,
            model_ig.m_seq_len: t_m_seq_len,
            model_ig.dropout_keep_prob:0.8
        }
        _, step, input, word_att, loss, prediction, y_g = sess.run(
            [train_op, global_step, model_ig.input_x, model_ig.word_attention,
             model_ig.loss, model_ig.prediction, model_ig.input_yg],
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}".format(time_str, step, loss))
        #if(step % 500 == 0):
            #print("{}: step {}, {}, y_g {}, loss_g {} ,{}, y_i {}, loss_i{}".format(time_str, step,"\n", y_g,loss_g,"\n",y_i, loss_i))
        #train_summary_writer.add_summary(summaries, step)
        return loss

    def dev_step(d_x_batch, d_x_batch_part, d_y_batch, d_y_batch_i, d_memory, d_memory_part, d_seq_length, d_m_seq_len):
        feed_dict = {
            model_ig.input_x: d_x_batch,
            model_ig.input_x_part: d_x_batch_part,
            model_ig.input_yg: d_y_batch,
            model_ig.input_yi: d_y_batch_i,
            model_ig.memory: d_memory,
            model_ig.input_memory_part: d_memory_part,
            model_ig.sequence_lengths: d_seq_length,
            model_ig.m_seq_len: d_m_seq_len,
            model_ig.dropout_keep_prob:1
        }
        step, input, sentence_att, word_att, loss, prediction, y_g = sess.run(
            [global_step, model_ig.input_x, model_ig.sentence_attention, model_ig.word_attention, model_ig.loss,
             model_ig.prediction, model_ig.input_yg],
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        # print("{}: step {}, loss {:g}".format(time_str, step, loss))
        #if writer:
        #    writer.add_summary(summaries, step)
        return loss, sentence_att, word_att, prediction

    for epoch in range(FLAGS.epochs):
        print('epoch: ', epoch)
        train = list(zip(train_data, train_labelg, train_labeli, train_memoryg, train_data_part, train_memoryg_part))
        random.shuffle(train)
        train_data, train_labelg, train_labeli, train_memoryg, train_data_part, train_memoryg_part = map(list, zip(*train))
        recoder = open("recoder_7_300.txt","a")

        t_x_batch, t_x_batch_part, t_y_batch, t_y_batch_i, t_memory, t_memory_part, t_seq_length, t_seq_length_m = prepare_batch(train_data, train_labelg,
                                        train_labeli, train_memoryg, train_data_part, train_memoryg_part, FLAGS.batch_size)
        for i in range(0, len(t_x_batch)):
            loss_t = train_step(t_x_batch[i], t_x_batch_part[i], t_y_batch[i], t_y_batch_i[i], t_memory[i],
                                t_memory_part[i], t_seq_length[i], t_seq_length_m[i])
            loss_train.append(loss_t)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % 500 == 0:
                print('****************************************')
                print("\nEvaluation:")
                recoder.write("\nEvaluation*****************************")
                dev = list(zip(dev_data, dev_labelg, dev_labeli, dev_memoryg, dev_data_part, dev_memoryg_part))
                random.shuffle(dev)
                dev_data, dev_labelg, dev_labeli, dev_memoryg, dev_data_part, dev_memoryg_part = map(list, zip(*dev))
                d_x_batch, d_x_batch_part, d_y_batch, d_y_batch_i, d_memory, d_memory_part, d_seq_length, d_seq_length_m = prepare_batch(dev_data, dev_labelg,
                                        dev_labeli, dev_memoryg, dev_data_part, dev_memoryg_part, FLAGS.batch_size)
                predictions = []
                labels = []
                loss_d_temp = []
                for j in range(0, len(d_x_batch)):
                    loss_d, sentence_att_d, word_att_d, each_pred = dev_step(d_x_batch[j], d_x_batch_part[j], d_y_batch[j], d_y_batch_i[j], d_memory[j],
                                                d_memory_part[j], d_seq_length[j], d_seq_length_m[j])
                    loss_d_temp.append(loss_d)
                    for k in range(0, len(each_pred)):
                        predictions.extend(each_pred[k,:d_seq_length[j][k]])
                        labels.extend(d_y_batch[j][k,:d_seq_length[j][k]])
                loss_dev.append(sum(loss_d_temp)/len(loss_d_temp))
                recoder.write(str(current_step)+"\n")
                recoder.write(str(sum(loss_d_temp)/len(loss_d_temp))+"\n")
                print(accuracy_score(labels, predictions))
                recoder.write(str(accuracy_score(labels, predictions))+"\n")
                print(classification_report(labels, predictions))
                recoder.write(str(classification_report(labels, predictions))+"\n")
                print(confusion_matrix(labels, predictions))
                recoder.write(str(confusion_matrix(labels, predictions))+"\n")
                print("*******************************")
                recoder.write("*******************************"+"\n")

                print("\nTest: ")
                recoder.write("Test************************************")
                test = list(zip(test_data, test_labelg, test_labeli, test_memoryg, test_data_part, test_memoryg_part))
                #random.shuffle(test)
                test_data, test_labelg, test_labeli, test_memoryg, test_data_part, test_memoryg_part = map(list, zip(*test))
                e_x_batch, e_x_batch_part, e_y_batch, e_y_batch_i, e_memory, e_memory_part, e_seq_length, e_seq_length_m = prepare_batch(test_data,
                                   test_labelg, test_labeli, test_memoryg, test_data_part, test_memoryg_part, FLAGS.batch_size)
                test_input = []
                test_memory = []
                test_sentence_attention = []
                test_word_attention = []
                test_prediction = []
                test_label = []
                test_predictions = []
                test_labels = []
                for l in range(0, len(e_x_batch)):
                    _,test_sentence_att, test_word_att, test_each_pred = dev_step(e_x_batch[l], e_x_batch_part[l], e_y_batch[l], e_y_batch_i[l], e_memory[l], e_memory_part[l], e_seq_length[l], e_seq_length_m[l])
                    for m in range(0, len(test_each_pred)):
                        test_predictions.extend(test_each_pred[m,:e_seq_length[l][m]])
                        test_labels.extend(e_y_batch[l][m,:e_seq_length[l][m]])
                    for d in range(0, len(test_each_pred)):
                        test_input.append(e_x_batch[l][d,:e_seq_length[l][d]])
                        test_memory.append(e_memory[l][d,:])
                        test_sentence_attention.append(test_sentence_att[d,:e_seq_length[l][d],:])
                        test_word_attention.append(test_word_att[d,:e_seq_length[l][d],:])
                        test_prediction.append(test_each_pred[d,:e_seq_length[l][d]])
                        test_label.append(e_y_batch[l][d,:e_seq_length[l][d]])
                if(current_step ==25000):
                    print("test_input",len(test_input))
                    print(test_input[0])
                    print(test_input[1])
                    print("test_memory",len(test_memory))
                    # print(type(test_word_attention))
                    print("test_sentence_attention",len(test_sentence_attention))
                    print(test_sentence_attention[0])
                    print("test_word_attention",len(test_word_attention))
                    print(test_word_attention[0])
                    # print(len(test_word_attention[0]))
                    # print(len(test_word_attention[1]))
                    # print(len(test_word_attention[0][0]))
                    print("test_prediction",len(test_prediction))
                    print("test_label",len(test_label))
                    test_input_word = []
                    test_memory_word = []
                    for items_test_input in test_input:
                        str_test_input = ""
                        for number in items_test_input:
                            get_index = dict_index.index(number)
                            str_test_input = str_test_input + dict_word[get_index]+ " "
                        test_input_word.append(str_test_input)
                    print("finish input")
                    print("test_input_word",len(test_input_word))
                    print("test_input_word[0]",test_input_word[0])

                    for items_test_memory in test_memory:
                        temp = []
                        for i in items_test_memory:
                            str_test_memory = ""
                            for index in i:
                                get_index = dict_index.index(index)
                                str_test_memory = (str_test_memory + dict_word[get_index]+ " ").replace("mask ","")
                            temp.append(str_test_memory)
                        str_temp = "\n".join(temp)
                        test_memory_word.append(str_temp)
                    print("test_memory_word",len(test_memory_word))
                    print("test_memory_word[0]",test_memory_word[0])

                    column1 = pd.Series(test_input_word, name='input')
                    column2 = pd.Series(test_memory_word, name='memory')

                    column3 = pd.Series(test_sentence_attention, name='attention')
                    column4 = pd.Series(test_prediction, name='prediction')
                    column5 = pd.Series(test_label, name='label')
                    save = pd.concat([column1, column2, column3, column4, column5], axis=1)
                    save.to_csv("result_7_300.csv", encoding="gbk")
                    f = open("result_7_300.txt", "a")

                    temp_count = 0
                    for case in test_word_attention:
                        str1 = str(temp_count) + "***" + "\n" + str(case) + "\n"
                        f.write(str1)
                        temp_count = temp_count + 1

                print(accuracy_score(test_labels, test_predictions))
                recoder.write(str(accuracy_score(test_labels, test_predictions))+"\n")
                print(classification_report(test_labels, test_predictions))
                recoder.write(str(classification_report(test_labels, test_predictions))+"\n")
                print(confusion_matrix(test_labels, test_predictions))
                recoder.write(str(confusion_matrix(test_labels, test_predictions))+"\n")
                print("\nTest Finished!!!")
                print("*******************************")
                recoder.write("*******************************"+"\n")
            #if current_step % FLAGS.checkpoint_every == 0:
            #    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            #    print("Saved model checkpoint to {}\n".format(path))

print(len(loss_train))
print(len(loss_dev))
plt.switch_backend('agg')
plt.xlabel('batch', fontproperties='SimHei', fontsize=20)
plt.ylabel('loss', fontproperties='SimHei', fontsize=20)
plt.figure(1)
plt.plot(loss_train, color="green",marker=".", label="train")
loss_dev_x = [500+500*i for i in range(len(loss_dev))]
plt.plot(list(loss_dev_x),loss_dev, color="red", marker=".", label="dev")
plt.legend(loc='upper left')
plt.savefig('loss_em150_lr_modifymask_pan1e-3',dpi=800)
