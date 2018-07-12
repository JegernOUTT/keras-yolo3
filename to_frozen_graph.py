import os
import keras
import keras.backend as K
from tensorflow.python.framework import graph_util, graph_io


os.environ["CUDA_VISIBLE_DEVICES"] = ""
K.set_learning_phase(0)

model_path = 'snapshots/current_heads/'

model = keras.models.load_model(os.path.join(model_path, 'yolo3_model.h5'))
model.summary()

output_names = [o.name.split(':')[0] for o in model.output]

sess = K.get_session()

constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), output_names)
print(str(constant_graph))
graph_io.write_graph(constant_graph, ".", os.path.join(model_path, 'pnet.pb'),
                     as_text=False)
