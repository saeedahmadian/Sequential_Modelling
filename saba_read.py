from Preprocess import *
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import copy
from sklearn.preprocessing import MinMaxScaler
tf.keras.backend.set_floatx('float64')
mypipline=Pipeline([('read_data',ReadData(data_dir='./data_new',file_name='cancer.csv')),
                    ('clean_data',CleanData('median'))
                    # ('cat_to_num',CatToNum('ordinal'))
                    # ,('Outlier_mitigation',OutlierDetection(threshold=2,name='whisker'))
                    ])

new_data= mypipline.fit_transform(None)
data_train,data_test = train_test_split(new_data,test_size=.3,random_state=8)

y_train_class= data_train.pop('G4RIL').values
y_train_reg= data_train.pop('CRT_ALCnadir')

y_test_class= data_test.pop('G4RIL').values
y_test_reg = data_test.pop('CRT_ALCnadir')


dense_features1= ['Age','BMI','CRT0ALC','Total_blood_volume_litres_Nadlerformula']
dense_features2= ['PTV', 'bodyV5_rel','bodyV10_rel','bodyV15_rel',
                'bodyV20_rel','bodyV25_rel','bodyV30_rel','bodyV35_rel','bodyV40_rel',
                'bodyV45_rel','bodyV50_rel','meanbodydose','bodyvolume','lungV5_rel',
                'lungV10_rel','lungV15_rel','lungV20_rel','lungV25_rel','lungV30_rel',
                'lungV35_rel','lungV40_rel','lungV45_rel','lungV50_rel','meanlungdose',
                'lungvolume','heartV5_rel','heartV10_rel','heartV15_rel','heartV20_rel',
                'heartV25_rel','heartV30_rel','heartV35_rel','heartV40_rel','heartV45_rel',
                'heartV50_rel','meanheartdose','heartvolume','spleenV5_rel','spleenV10_rel',
                'spleenV15_rel','spleenV20_rel','spleenV25_rel','spleenV30_rel','spleenV35_rel',
                'spleenV40_rel','spleenV45_rel','spleenV50_rel','meanspleendose','spleenvolume'
                  ]

dense_features= dense_features1+dense_features2
sparse_features = ['IMRT1Protons0','Sex','Race','Histology',
                   'Location_uppmid_vs_low','Location_upp_vs_mid_vs_low','Induction_chemo',
                   'CChemotherapy_type']
sequential_features_t0 = [
    'CRT0neutrophil_percent','CRT0lymphocyte_percent','CRT0monocyte_percent'
    ]
# sequential_features_t1=[
#     'CRT1Red_blood_cell_MuL','CRT1hemoglobin_GDL','CRT1Hematocrit','CRT1White_blood_cell_KuL',
#     'CRT1neutrophil_absolute_count_KuL','CRT1neutrophil_percent','CRT1lymphocyte_absolute_count_KuL',
#     'CRT1lymphocyte_percent','CRT1neutrophiltolymphocyte_ratio','CRT1monocyte_absolute_count_KuL',
#     'CRT1monocyte_percent','CRT1eosinophil_absolute_count_KuL','CRT1eosinophil_percent',
#     'CRT1basophil_absolute_count_KuL','CRT1basophil_percent'
#
# ]
#
# sequential_features_t2= [
#     'CRT2Red_blood_cell_MuL','CRT2hemoglobin_GDL','CRT2Hematocrit','CRT2White_blood_cell_KuL',
#     'CRT2neutrophil_absolute_count_KuL','CRT2neutrophil_percent','CRT2lymphocyte_absolute_count_KuL',
#     'CRT2lymphocyte_percent','CRT2neutrophiltolymphocyte_ratio','CRT2monocyte_absolute_count_KuL',
#     'CRT2monocyte_percent','CRT2eosinophil_absolute_count_KuL','CRT2eosinophil_percent',
#     'CRT2basophil_absolute_count_KuL','CRT2basophil_percent'
# ]
#
# sequential_features_t3 = [
#     'CRT3Red_blood_cell_MuL', 'CRT3hemoglobin_GDL', 'CRT3Hematocrit', 'CRT3White_blood_cell_KuL',
#     'CRT3neutrophil_absolute_count_KuL', 'CRT3neutrophil_percent', 'CRT3lymphocyte_absolute_count_KuL',
#     'CRT3lymphocyte_percent', 'CRT3neutrophiltolymphocyte_ratio', 'CRT3monocyte_absolute_count_KuL',
#     'CRT3monocyte_percent', 'CRT3eosinophil_absolute_count_KuL', 'CRT3eosinophil_percent',
#     'CRT3basophil_absolute_count_KuL', 'CRT3basophil_percent'
# ]
#
# sequential_features_t4=[
#     'CRT4Red_blood_cell_MuL', 'CRT4hemoglobin_GDL', 'CRT4Hematocrit', 'CRT4White_blood_cell_KuL',
#     'CRT4neutrophil_absolute_count_KuL', 'CRT4neutrophil_percent', 'CRT4lymphocyte_absolute_count_KuL',
#     'CRT4lymphocyte_percent', 'CRT4neutrophiltolymphocyte_ratio', 'CRT4monocyte_absolute_count_KuL',
#     'CRT4monocyte_percent', 'CRT4eosinophil_absolute_count_KuL', 'CRT4eosinophil_percent',
#     'CRT4basophil_absolute_count_KuL', 'CRT4basophil_percent'
# ]
#
# sequential_features_t5 = [
#     'CRT5Red_blood_cell_MuL', 'CRT5hemoglobin_GDL', 'CRT5Hematocrit', 'CRT5White_blood_cell_KuL',
#     'CRT5neutrophil_absolute_count_KuL', 'CRT5neutrophil_percent', 'CRT5lymphocyte_absolute_count_KuL',
#     'CRT5lymphocyte_percent', 'CRT5neutrophiltolymphocyte_ratio', 'CRT5monocyte_absolute_count_KuL',
#     'CRT5monocyte_percent', 'CRT5eosinophil_absolute_count_KuL', 'CRT5eosinophil_percent',
#     'CRT5basophil_absolute_count_KuL', 'CRT5basophil_percent'
# ]
#
# sequential_features_t6 = [
#     'CRT6Red_blood_cell_MuL', 'CRT6hemoglobin_GDL', 'CRT6Hematocrit', 'CRT6White_blood_cell_KuL',
#     'CRT6neutrophil_absolute_count_KuL', 'CRT6neutrophil_percent', 'CRT6lymphocyte_absolute_count_KuL',
#     'CRT6lymphocyte_percent', 'CRT6neutrophiltolymphocyte_ratio', 'CRT6monocyte_absolute_count_KuL',
#     'CRT6monocyte_percent', 'CRT6eosinophil_absolute_count_KuL', 'CRT6eosinophil_percent',
#     'CRT6basophil_absolute_count_KuL', 'CRT6basophil_percent'
# ]

sequential_features_t1=['CRT1lymphocyte_absolute_count_KuL','CRT1lymphocyte_percent']
sequential_features_t2=['CRT2lymphocyte_absolute_count_KuL','CRT2lymphocyte_percent']
sequential_features_t3=['CRT3lymphocyte_absolute_count_KuL','CRT3lymphocyte_percent']
sequential_features_t4=['CRT4lymphocyte_absolute_count_KuL','CRT4lymphocyte_percent']
sequential_features_t5=['CRT5lymphocyte_absolute_count_KuL','CRT5lymphocyte_percent']


sequential_features= sequential_features_t1+sequential_features_t2+\
                     sequential_features_t3+sequential_features_t4+\
                     sequential_features_t5


class saba_class(tf.keras.models.Model):
    def __init__(self,hidd_size=20,lstm_size=20,max_seq=5,**kwargs):
        super(saba_class,self).__init__(**kwargs)
        self.max_seq= max_seq
        self.hidd_size= hidd_size
        self.lstm_size= lstm_size
        self.dense_layer= tf.keras.layers.Dense(units=20,activation='relu',name='Dense_input',dtype=tf.float64)
        self.sparse_layer= tf.keras.layers.Dense(units=20,
                                                 kernel_regularizer=tf.keras.regularizers.l1(.2),
                                                 name='sparse_input',dtype=tf.float64)
        self.init_seq= tf.keras.layers.Dense(units=10,activation='relu',name='initial_sequence',dtype=tf.float64)
        self.combine_layer= tf.keras.layers.Dense(units=self.hidd_size,activation='relu',
                                                  name='combined_layer',dtype=tf.float64)
        self.lstm_layer_list= [tf.keras.layers.LSTM(units=self.lstm_size,return_sequences=True,
                                                    return_state=True,name='lstm_cell_{}'.format(i),dtype=tf.float64)
                               for i in range(self.max_seq)]
        # self.lstm_layer= tf.keras.layers.LSTM(units=self.lstm_size,return_sequences=True,return_state=True)
        # self.last_layer = tf.keras.layers.Dense(units=1,activation='sigmoid',name='output_layer',dtype=tf.float64)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None,53],dtype=tf.float64,name='input_dense'),
                                  tf.TensorSpec(shape=[None,8],dtype=tf.float64,name='input_sparse'),
                                  tf.TensorSpec(shape=[None,3],dtype=tf.float64,name='input_init_seq'),
                                  tf.TensorSpec(shape=[None,2],dtype=tf.float64,name='initial_state_1'),
                                  tf.TensorSpec(shape=[None, 2], dtype=tf.float64, name='initial_state_2')
                                  ])
    def call(self,x_dense,x_sparse,x_init_seq,initial_state1,initial_state2):
        x_dense= self.dense_layer(x_dense)
        x_sparse= self.sparse_layer(x_sparse)
        x_init_seq= self.init_seq(x_init_seq)
        x_proc= self.combine_layer(tf.concat([x_dense,x_sparse,x_init_seq],axis=-1))
        out= tf.expand_dims(x_proc,1)
        seq_output = []
        i=0
        init_states=[initial_state1,initial_state2]
        for lstm in self.lstm_layer_list:
            out, hiddent_state, cell_state = lstm(out, initial_state=init_states)
            seq_output.append(out)
            init_states=[hiddent_state,cell_state]
            i+=1
        # out= self.last_layer(tf.reshape(out,shape=[-1,self.lstm_size]))
        return out, seq_output

    def get_config(self):
        config = super(saba_class,self).get_config().copy()
        config.update({
            'maximum_length_seq': self.max_seq,
            'hidden_size': self.hidd_size,
            'LSTM_size': self.lstm_size
        })
        return config


def string_float(data):
    df= copy.deepcopy(data)
    columns= data.shape[1]
    for col in range(columns):
        tmp=list(map(lambda x: 0 if x==' ' else float(x),data.iloc[:,col].tolist()))
        median= np.median(tmp)
        df.iloc[:,col]= list(map(lambda x: median if x==0 else x,tmp))
    return df

MAX_SEQ = 5
epochs = 1
batch_size= 32
NUM_SEQ_Features= 2
optim= tf.keras.optimizers.Adam(learning_rate=1e-3,name='adam_optim')
model= saba_class(lstm_size=NUM_SEQ_Features)


x_train_dense = MinMaxScaler((0,1)).fit_transform(string_float(data_train[dense_features]).values)
x_train_sparse= MinMaxScaler((0,1)).fit_transform(string_float(data_train[sparse_features]).values)
x_train_init_seq=MinMaxScaler((0,1)).fit_transform(string_float(data_train[sequential_features_t0]).values)
y_train_sequential = MinMaxScaler((0,1)).fit_transform(string_float(data_train[sequential_features]).values)
y_train_class


dataset = tf.data.Dataset.\
    from_tensor_slices((x_train_dense,x_train_sparse,x_train_init_seq,
                        y_train_sequential,y_train_class)).batch(batch_size)

checkpoint_dir = './check_points'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optim,
                                 model=model)

loss_pred= tf.keras.losses.MeanSquaredError(name='mse')
loss_class = tf.keras.losses.BinaryCrossentropy(name='cross_entropy')
def train_step(x_dense,x_sparse,x_init_seq,init_state1,init_state2,seq_target,
               class_target):
    loss_total=0
    with tf.GradientTape() as tape:
        y_label,output_sequence=model(x_dense,x_sparse,x_init_seq,init_state1,init_state2)
        for i,out in enumerate(output_sequence):
            y_targ= seq_target[:,i*NUM_SEQ_Features:(i+1)*NUM_SEQ_Features]
            loss_total+=loss_pred(y_targ,tf.reshape(out,shape=[-1,NUM_SEQ_Features]))
        # loss_total+=loss_class(class_target,y_label)

    gradients= tape.gradient(loss_total, model.trainable_variables)
    optim.apply_gradients(zip(gradients, model.trainable_variables))

    return loss_total

def myplot(y_true,y_pred,nrows=5,ncols=5,title='CRT1lymphocyte_absolute_count_KuL'):
    fig,axes= plt.subplots(nrows,ncols,sharex='none',sharey='none',figsize=(20,20))
    seq_len= y_true.shape[1]
    i=0
    for row_ax in axes:
        for col_ax in row_ax:
            col_ax.plot(np.arange(seq_len),y_true[i,:],color='darkblue',label='True_value')
            col_ax.plot(np.arange(seq_len),y_pred[i,:],color='darkorange',label='Pred_value')
            col_ax.legend()
            i+=1
    fig.savefig('./Figs/fig_{}.png'.format(title))

train_loss=[]
for epoch in range(epochs):
    print('epoch {}/{} starts...'.format(epoch,epochs))
    dataset= dataset.shuffle(600)
    i=0
    for x_batch_dense,x_batch_sparse,x_batch_init_seq,y_batch_sequential,y_batch_class in dataset:
        current_batch= x_batch_dense.shape[0]
        initial_state1= tf.zeros((current_batch,NUM_SEQ_Features),dtype=tf.float64)
        initial_state2= tf.zeros((current_batch,NUM_SEQ_Features),dtype=tf.float64)
        # initial_state= [tf.zeros((current_batch,NUM_SEQ_Features),dtype=tf.float64),tf.zeros((current_batch,NUM_SEQ_Features),dtype=tf.float64)]
        batch_loss=train_step(x_batch_dense, x_batch_sparse,
                              x_batch_init_seq, initial_state1,initial_state2,
                              y_batch_sequential,y_batch_class)
        train_loss.append(batch_loss)

        if i % 5 ==0:
            checkpoint.save(file_prefix=checkpoint_prefix)
            model.save_weights('manual_checkpoint/mymodel-{}-ckpt'.format(i))
            print('epoch {}/{} iter {} loss is {}'.format(epoch,epochs,i,batch_loss))
        i+=1

    # print('Loss value for epoch {} is {}'.format(epoch,batch_loss))


test_mode = True
if test_mode== True:
    x_test_dense = MinMaxScaler((0, 1)).fit_transform(string_float(data_test[dense_features]).values)
    x_test_sparse = MinMaxScaler((0, 1)).fit_transform(string_float(data_test[sparse_features]).values)
    x_test_init_seq = MinMaxScaler((0, 1)).fit_transform(string_float(data_test[sequential_features_t0]).values)
    y_test_sequential = MinMaxScaler((0, 1)).fit_transform(string_float(data_test[sequential_features]).values)
    current_batch = x_test_dense.shape[0]
    init_state= [tf.zeros((current_batch,NUM_SEQ_Features),dtype=tf.float64),tf.zeros((current_batch,NUM_SEQ_Features),dtype=tf.float64)]
    y_label, output_sequence = model(x_test_dense, x_test_sparse, x_test_init_seq, init_state)
    RMSE = []
    for i, out in enumerate(output_sequence):
        y_targ = y_test_sequential[:, i * NUM_SEQ_Features:(i + 1) * NUM_SEQ_Features]
        out_ = tf.reshape(out, [-1, NUM_SEQ_Features])
        RMSE.append(tf.reduce_mean(tf.keras.metrics.mean_squared_error(y_targ,out_)))
    output_sequence_= [tf.reshape(out, [-1, NUM_SEQ_Features]) for out in output_sequence]
    output_sequence_1= tf.concat([tf.expand_dims(out[:,0],-1) for out in output_sequence_],axis=-1)
    output_sequence_2 = tf.concat([tf.expand_dims(out[:,1],-1) for out in output_sequence_],axis=-1)
    y_test_sequential_1= y_test_sequential[:,[i for i in range(0,10,2) ]]
    y_test_sequential_2 = y_test_sequential[:, [i for i in range(1, 10, 2)]]
    print('Start to plot figures')
    myplot(y_test_sequential_1,output_sequence_1,5,5,'CRT1lymphocyte_absolute_count_KuL')
    myplot(y_test_sequential_2,output_sequence_2,5,5,'CRT1lymphocyte_percent')
    a = 1

        # for j in range(NUM_SEQ_Features):













