from functions import *
from params import *
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

train_csv = pd.read_csv("ndsc-beginner/train.csv")

_X = np.array(train_csv['image_path'].values)

X=[]

num=1000
for i in tqdm(range(0,len(_X))):
        if i<num:
                img = image.load_img(_X[i], target_size=(224, 224))
                x = image.img_to_array(img)
                # x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                X.append(x)

X = np.array(X)
y = np.array(train_csv['Category'].values)
print (np.array(X).shape)
print (np.array(y).shape)

# encode the label into binary arrays
le = LabelEncoder()
y = to_categorical(y[:num], le, fit=True)
# np.save(out_dir+'/class_num',len(le.classes_))
# print (le.classes_)
print('TOTAL NUM OF LABELS : %d'%(len(le.classes_)))
num_classes=len(le.classes_)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 
print (X_train.shape)
print (X_test.shape)
print (y_train.shape)
print (y_test.shape)

with tf.Session('') as sess:
        # KTF.set_session(sess)
        # sess.run(tf.global_variables_initializer())
        # sess = tf.Session(config=tf.ConfigProto(device_count={'gpu':3}))

        KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu':3})))
        ################# CREATE MODEL ##################
        base_model = VGG19(input_shape=(224,224,3),weights='imagenet', include_top=False)
        for layer in base_model.layers:
                layer.trainable = False
        x = base_model.output
        x=GlobalAveragePooling2D()(x)
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
        model.summary()

        history = model.fit(X_train, y_train,
                        epochs=10,
                        verbose=True,
                        validation_data=(X_test, y_test),
                        batch_size=64)

        loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
        print("Training Accuracy: {:.4f}".format(accuracy))
        loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
        print("Testing Accuracy:  {:.4f}".format(accuracy))

        # evaluation model
        preds = model.predict(X_test)
        preds = np.argmax(preds,axis=1)
        print (np.array(preds).shape)


        # Compute confusion matrix
        from sklearn.metrics import confusion_matrix
        _y_test = np.argmax(y_test, axis = 1)
        print (np.array(_y_test).shape)
        cnf_matrix = confusion_matrix(_y_test, preds)
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plot_confusion_matrix(cnf_matrix, classes=le.classes_, savefig='confusion_matrix_1')