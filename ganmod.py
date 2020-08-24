import tensorflow as tf
import readData
import numpy as np
from sklearn.utils import shuffle
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report, roc_auc_score
import matplotlib.pylab as Plot
import matplotlib
import matplotlib.pyplot as plt

X_dim = 27


def step_targeted_attack(X, eps, target , logits):
    #X = tf.placeholder(tf.float32, [None, X_dim])
    one_hot_target_class = tf.one_hot(target, 1)
    #print(one_hot_target_class,"\n\n")
    cross_entropy = tf.losses.softmax_cross_entropy(one_hot_target_class,logits,label_smoothing=0.1,weights=1.0)
    print(tf.sign(tf.gradients(cross_entropy,X)[0]).shape)
    x_adv = X - eps * tf.sign(tf.gradients(cross_entropy, X)[0])
    x_adv = tf.clip_by_value(x_adv, 0, 1.0)
    return tf.stop_gradient(x_adv)

def get_batch(X_train, y_train, batchsize):
    X, y = shuffle(X_train, y_train)
    count_0 = 0
    count_1 = 0
    ind = list()
    count = count_0 + count_1
    i = 0
    while(count<batchsize):
        if y[i] == 0:
            if count_0 < batchsize/2:
                ind.append(i)
                count_0 = count_0+1
        elif y[i] == 1:
            if count_1 < batchsize/2:
                ind.append(i)
                count_1 = count_1 + 1
        count = count_1 + count_0
        i = i+1
    ind = np.array(ind)
    X_batch = X[ind]
    #X_batch = X[:batchsize,:]
    #y_batch = y[:batchsize]
    y_batch = y[ind]
    return X_batch, y_batch


def sample_Z(batchsize):
    z = np.random.standard_normal((batchsize, 2)).astype(np.float32)
    return z


def generator(Z,hsize=[16, 16],reuse=False):
    with tf.variable_scope("GAN/Generator",reuse=reuse):
        h1 = tf.layers.dense(Z,hsize[0],activation=tf.nn.leaky_relu)
        h2 = tf.layers.dense(h1,hsize[1],activation=tf.nn.leaky_relu)
        out = tf.layers.dense(h2,X_dim)
    return out


def discriminator(X,hsize=[16, 16],reuse=False):
    with tf.variable_scope("GAN/Discriminator",reuse=reuse):
        h1 = tf.layers.dense(X,hsize[0],activation=tf.nn.leaky_relu)
        h2 = tf.layers.dense(h1,hsize[1],activation=tf.nn.leaky_relu)
        h3 = tf.layers.dense(h2,2)
        out = tf.layers.dense(h3,1)
    return out, h3


eps = 0.0005


X = tf.placeholder(tf.float32,[None,X_dim])
Z = tf.placeholder(tf.float32,[None,2])
Y = tf.placeholder(tf.float32,[None,1])
X_adver = tf.placeholder(tf.float32,[None,X_dim])
Y_adver = tf.placeholder(tf.float32,[None,1])


G_sample = generator(Z)
r_logits, r_rep = discriminator(X)
f_logits, g_rep = discriminator(G_sample,reuse=True)

#disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logits,labels=tf.ones_like(r_logits)) + tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits,labels=tf.zeros_like(f_logits)))
disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logits,labels=Y) + tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits,labels=tf.ones_like(f_logits)))
gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits,labels=tf.zeros_like(f_logits)))


gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Generator")
disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Discriminator")

gen_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(gen_loss,var_list = gen_vars) # G Train step
disc_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(disc_loss,var_list = disc_vars) # D Train step
pred_log = tf.sigmoid(r_logits)
pred_y = tf.cast(pred_log+0.15, tf.int32)
pred_y = tf.cast(pred_y, tf.float32)
actual = Y
correct_pred = tf.equal(pred_y, Y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

adv_logits, _ = discriminator(X_adver,reuse=True)

#one_hot_target_class = tf.one_hot(target, 1)
    #print(one_hot_target_class,"\n\n")
adv_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y_adver,logits=adv_logits)
#print(tf.sign(tf.gradients(adv_cross_entropy,X_adv)[0]).shape)
#gradients = tf.sign(tf.gradients(adv_cross_entropy, X_adver)[0])
gradients = tf.gradients(adv_cross_entropy, X_adver)[0]
gradient_abs = tf.math.abs(gradients)
gradient_max = tf.argmax(tf.reshape(gradient_abs, [27]),0)
signed_gradients = tf.sign(gradients)[0]
temp = tf.Variable(tf.ones_like(signed_gradients))
temp = temp[gradient_max].assign(tf.constant(0.0))
#signed_gradients = tf.multiply(signed_gradients, temp) #uncomment this for limiting
#signed_gradients = signed_gradients[gradient_max].assign(tf.constant(0.0))
x_adv = X_adver - eps * signed_gradients
x_adv = tf.clip_by_value(x_adv, 0, 1.0)
x_adv = tf.stop_gradient(x_adv)
    #return tf.stop_gradient(x_adv)


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)

batch_size = 50
X_train, y_train, X_test, y_test, dictionary, renorm = readData.getdata()
init_op = tf.initialize_all_variables()
list_adversarial_success = list()

y_train = np.reshape(y_train,newshape=[y_train.shape[0], 1])
y_test = np.reshape(y_test,newshape=[y_test.shape[0], 1])
saver = tf.train.Saver()
most_sig_grad = {}
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init_op)
    for i in range(50000):
        X_batch, y_batch = get_batch(X_train, y_train, batch_size)
        #print(y_batch)
        Z_batch = sample_Z(batch_size)
        _, dloss = sess.run([disc_step, disc_loss], feed_dict={X: X_batch, Y:y_batch, Z: Z_batch})
        _, gloss = sess.run([gen_step, gen_loss], feed_dict={Z: Z_batch})
        if i%100 == 0:
            acc, predicted_y = sess.run([accuracy, pred_y], feed_dict={X: X_test, Y: y_test})
            print('iter {} dloss {} gloss {} test acc {}'.format(i, dloss, gloss, acc))

        if i%2000 == 0:
            acc, predicted_y = sess.run([accuracy, pred_y], feed_dict={X: X_test, Y: y_test})
            print('{}'.format(classification_report(y_test, predicted_y)))
            print('roc auc score {}'.format(roc_auc_score(y_test, predicted_y)))
            norm_inds = list()
            ano_inds = list()
            for i in range(y_train.shape[0]):
                if y_train[i] == 0:
                    norm_inds.append(i)
                elif y_train[i] == 1:
                    ano_inds.append(i)

            X_train_normal = X_train[norm_inds]
            X_train_ano = X_train[ano_inds]
            list_adv_before = list()
            list_adv_after = list()

            for k in range(50):
                #X_t_normal = np.random.shuffle(X_train_normal)
                X_attack = X_train_normal[k,:]
                X_adv = np.reshape(X_attack, (1,X_dim))
                #print(X_adv.shape)
                #X_adv = np.expand_dims(X_attack, 0)
                #X_adv = X_attack
                Y_att = 0
                Y_att = np.array(Y_att)
                Y_att = np.reshape(Y_att, (1,1))
                Y_target = 1
                Y_target = np.array(Y_target)
                Y_target = np.reshape(Y_target, (1,1))
                #print('old X {}'.format(X_adv))
                before_pred, logits = sess.run([pred_y, r_logits], feed_dict={X: X_adv, Y: Y_att})

                for j in range(3000):
                    #print(j)
                    X_adv = np.reshape(X_adv, (1,X_dim))
                    X_adv, grad_max, grads, grad_abs = sess.run([x_adv, gradient_max, gradients, signed_gradients], feed_dict={X_adver: X_adv, Y_adver: Y_target})
                    if grad_max in most_sig_grad:
                        most_sig_grad[grad_max] = most_sig_grad[grad_max] + 1
                    else:
                        most_sig_grad[grad_max] = 0
                    #print('gradmax {} gradients {} signed shape {} shape {}'.format(gradmax, grads, grad_abs.shape, grads.shape))
                    #print('gradients {}'.format(grads))
                    #attack_pred, logits = sess.run([pred_y, r_logits], feed_dict={X: X_adv, Y: Y_att})
                    #print('attack pred {} logits {}'.format(attack_pred,logits))
                    #X_adv = step_targeted_attack(X_adv,0.005,Y_target,logits)
                X_adv = np.reshape(X_adv, (1, X_dim))
                #print('new X {}'.format(X_adv))
                after_pred = sess.run([pred_y], feed_dict={X:X_adv, Y:Y_target})
                list_adv_before.append(before_pred[0][0])
                list_adv_after.append(after_pred[0][0])
                #after_pred, logits = sess.run([accuracy, r_logits], feed_dict={X: X_adv, Y: Y_att})
                #print('adversarial attack before predict {} after predict {}'.format( before_pred, after_pred))
            #print('list before {} list after {}'.format(list_adv_after, list_adv_before))
            list_adversarial_success.append(1-accuracy_score(list_adv_before, list_adv_after))
            print('adv success percentage {}'.format(1-accuracy_score(list_adv_before, list_adv_after)))
            for key in most_sig_grad.keys():
                print('most_sig_grad for {} support {}'.format(key, most_sig_grad[key]))
            #display_z = sample_Z(batch_size)
            f, (ax1, ax2, ax3) = Plot.subplots(1, 3)
            ax1.set_autoscale_on(False)
            ax2.set_autoscale_on(False)
            Zbatch = sample_Z(1500)
            gen = sess.run([G_sample], feed_dict={Z:Zbatch})
            gen = np.array(gen)
            ax1.scatter(gen[:, 0], gen[:, 1], s=20)
            # ax1.scatter(gen[lab1,0], gen[lab1,1], color='r');			# Uncomment this line when testing with multimodal data
            ax1.set_title('Generated samples')
            ax1.set_aspect('equal')
            ax1.axis([-1, 1, -1, 1])
            X_t = np.array(X_train)
            Y_t = np.array(y_train)
            Y_t = np.reshape(Y_t, newshape=(Y_t.shape[0]))
            colors = ['blue', 'red']
            ax2.scatter(X_t[:, 0], X_t[:, 1], c=Y_t, cmap=matplotlib.colors.ListedColormap(colors), s=20)
            ax2.set_title('Training samples')
            ax2.set_aspect('equal')
            ax2.axis([-1, 1, -1, 1])
            inds = np.where(y_train > 0.0)
            # print(inds)
            X_anomaly = X_train[inds, :]
            X_anomaly = X_anomaly[1, :, :]
            #print('Attack samples {} {} {} {}'.format(X_anomaly.shape, X_anomaly[:,0], X_anomaly[:,1], y_train[inds]))
            ax3.scatter(X_anomaly[:, 0], X_anomaly[:, 1], s=20)
            ax3.set_title('Attack samples')
            ax3.set_aspect('equal')
            ax3.axis([-1, 1, -1, 1])
            f.savefig(str(i) + ".png")
            #saver.save(sess, 'C:\\Users\\Administrator\\PycharmProjects\\GANsec\\savedmodels\\')
    acc, predicted_y = sess.run([accuracy, pred_y], feed_dict={X:X_test, Y:y_test})
    print('accuracy on test {}'.format(acc))
    print('{}'.format(classification_report(y_test, predicted_y)))
    print('roc auc score {}'.format(roc_auc_score(y_test, predicted_y)))
    z_sample_batch = sample_Z(10000)
    generated_samples = sess.run([G_sample],  feed_dict={Z: z_sample_batch})
generated_samples = np.array(generated_samples)
print(generated_samples.shape[0])
generated_samples = np.reshape(generated_samples,newshape=[10000, X_dim])

norm_inds = list()
ano_inds = list()
for i in range(y_train.shape[0]):
    if y_train[i] == 0:
        norm_inds.append(i)
    elif y_train[i] == 1:
        ano_inds.append(i)

X_train_normal = X_train[norm_inds]
X_train_ano = X_train[ano_inds]

clf = IsolationForest(behaviour='new', max_samples=1500,
                           contamination='auto')
clf.fit(X_train_normal)
y_pred_gen = clf.predict(generated_samples)
y_pred_test = clf.predict(X_test)

print('shapes test {} generated {}'.format(X_test.shape, generated_samples.shape))
print(y_pred_test)
print(y_pred_gen)

y_pred_gen = np.array(y_pred_gen)
print(y_pred_gen.shape)
y_actual_gen = list()
for i in range(y_pred_gen.shape[0]):
    y_actual_gen.append(1)
    '''
    if y_pred_gen[i] == -1:
        y_pred_gen[i] = 1
    elif y_pred_gen[i] == 1:
        y_pred_gen = 0
    '''
print('adversarial success {}'.format(list_adversarial_success))
y_pred_gen = y_pred_gen + 1
y_pred_gen = y_pred_gen/2
y_pred_gen = 1 - y_pred_gen

for i in range(y_pred_test.shape[0]):
    if y_pred_test[i] == -1:
        y_pred_test[i] = 1
    elif y_pred_test[i] == 1:
        y_pred_test[i] = 0

print(y_pred_gen)
y_test_reshape = y_test.tolist()
y_pred_gen = y_pred_gen.tolist()

print('classification for test')
print(classification_report(y_test_reshape,y_pred_test))
print('classification for gen')
print(classification_report(y_actual_gen,y_pred_gen))

inds = np.where(y_train>0.0)
#print(inds)
X_anomaly = X_train[inds,:]
X_anomaly = X_anomaly[1,:,:]
inds = np.where(y_test<1.0)
#print(inds)
X_norm = X_test[inds,:]
X_norm = X_norm[1,:,:]
#X_anomaly = np.reshape(X_anomaly, (X_anomaly.shape[1], X_anomaly.shape[2]))
print(X_norm.shape)
for i in range(generated_samples.shape[1]):
    plt.clf()
    x= X_train[:,i]
    x_a = X_anomaly[:,i]
    x_b = X_norm[:,i]/10
    y = generated_samples[:,i]
    bins = np.linspace(0,1,10)
    #fig = plt.hist(x, bins, alpha=0.5, label='training')
    fig = plt.hist(y, bins, alpha=0.5, label='generated')
    fig = plt.hist(x_a, bins, alpha=0.5, label='attack')
    fig = plt.hist(x_b, bins, alpha=0.5, label='non-attack')
    plt.legend(loc='upper right')
    fname = 'hist' + str(i) + '.png'
    plt.savefig(fname)



f = open("gen_data.txt", "w+")
for i in range(generated_samples.shape[0]):
    gen_data = ""
    flag = 0
    for j in range(generated_samples.shape[1]):
        new_val = int(round(generated_samples[i,j]*renorm[j]))
        if new_val < 0:
            flag = 1
        key = (j, new_val)
        #print('generated sample data {} normalized data {} j {} key {} '.format(generated_samples[i, j], new_val, j, key))
        if key in dictionary:
            new_val = dictionary[key]
        #print('generated sample data {} normalized data {} j {} key {} '.format(generated_samples[i, j], new_val, j, key))
        gen_data = gen_data + ' ' + str(new_val)
    if flag == 0:
        #print('string to print {}'.format(gen_data))
        f.write(str(gen_data))
        f.write(' \t ')
        f.write(str(y_pred_gen[i]))
        f.write('\n')
    flag = 0

loss_object = tf.keras.losses.CategoricalCrossentropy()

def get_normal_sample():
    inds = np.where(y_train < 1.0)
    # print(inds)
    X_normal = X_train[inds, :]
    X_normal = X_normal[1, :, :]
    return X_normal[0,:]

'''
def create_adversarial_pattern(input_image, input_label):
  with tf.GradientTape() as tape:
    tape.watch(input_image)
    prediction = fee
    loss = loss_object(input_label, prediction)

  # Get the gradients of the loss w.r.t to the input image.
  gradient = tape.gradient(loss, input_image)
  # Get the sign of the gradients to create the perturbation
  signed_grad = tf.sign(gradient)
  return signed_grad

epsilons = [0, 0.01, 0.1, 0.15]
descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input')
                for eps in epsilons]

for i, eps in enumerate(epsilons):
  adv_x = image + eps*perturbations
  adv_x = tf.clip_by_value(adv_x, 0, 1)
  display_images(adv_x, descriptions[i])
'''