import keras
import keras.backend as K
from tensorflow.python.framework import graph_util, graph_io

K.set_learning_phase(0)

model = keras.models.load_model('snapshots_person_final/yolo3_model.h5')
model.summary()

output_names = [o.name.split(':')[0] for o in model.output]

sess = K.get_session()

constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), output_names)
graph_io.write_graph(constant_graph, ".", "pnet.pb", as_text=False)
