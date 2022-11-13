from tensorflow import keras
import os

"""
if os.path.exists("./swagger_server/models/workpieceStorage_marker_model.hdf5"):
    MARKER_MODEL_INIT = keras.models.load_model("./swagger_server/models/workpieceStorage_marker_model.hdf5")
else:
    MARKER_MODEL_INIT = keras.models.load_model("workpieceStorage_marker_model.hdf5")
"""

if os.path.exists("./swagger_server/models/workpieceStorage_marker_model_small.hdf5"):
    json_model_file = open("./swagger_server/models/workpieceStorage_marker_model_small.json", 'r')
    json_model = json_model_file.read()
    json_model_file.close()
    MARKER_MODEL_INIT = keras.models.model_from_json(json_model)
    MARKER_MODEL_INIT.load_weights("./swagger_server/models/workpieceStorage_marker_model_small.hdf5")
    MARKER_MODEL_INIT.compile(loss='binary_crossentropy', optimizer='adam', metrics=['loss', 'accuracy'])
else:
    json_model_file = open("models/workpieceStorage_marker_model_small.json", 'r')
    json_model = json_model_file.read()
    json_model_file.close()
    MARKER_MODEL_INIT = keras.models.model_from_json(json_model)
    MARKER_MODEL_INIT.load_weights("models/workpieceStorage_marker_model_small.hdf5")
    MARKER_MODEL_INIT.compile(loss='binary_crossentropy', optimizer='adam', metrics=['loss', 'accuracy'])

SHOW_PLOTS = True
