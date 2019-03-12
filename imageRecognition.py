
import os

from imageai.Prediction import ImagePrediction
from imageai.Prediction.Custom import ModelTraining

# obtains the path to the folder that contains your python file !!!!
execution_path = os.getcwd()

SOURCE_PATH = "https://github.com/OlafenwaMoses/IdenProf/releases/download/v1.0/idenprof-jpg.zip"
FILE_DIR = os.path.join(execution_path, "idenprof-jpg.zip")


model_trainer = ModelTraining()
model_trainer.setModelTypeAsResNet()
model_trainer.setDataDirectory("idenprof")
model_trainer.trainModel(num_objects=10, num_experiments=1, batch_size=32)
