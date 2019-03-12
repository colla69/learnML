
import os

from imageai.Prediction import ImagePrediction
from imageai.Prediction.Custom import ModelTraining

# obtains the path to the folder that contains your python file !!!!
execution_path = os.getcwd()

SOURCE_PATH = "https://github.com/OlafenwaMoses/IdenProf/releases/download/v1.0/idenprof-jpg.zip"
FILE_DIR = os.path.join(execution_path, "idenprof-jpg.zip")


# model_trainer = ModelTraining()
# model_trainer.setModelTypeAsResNet()
# model_trainer.setDataDirectory("idenprof")
# model_trainer.trainModel(num_objects=10, num_experiments=1, batch_size=32)

prediction = ImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath(os.path.join(execution_path, "idenprof/models/model_ex-001_acc-0.100000.h5"))

prediction.loadModel()


all_images_array = []


all_files = os.listdir(execution_path+"test")
for each_file in all_files:
    if each_file.endswith(".jpg") or each_file.endswith(".png"):
        all_images_array.append(each_file)
results_array = prediction.predictMultipleImages(all_images_array, result_count_per_image=5)
for each_result in results_array:
    predictions, percentage_probabilities = each_result["predictions"], each_result["percentage_probabilities"]
    for index in range(len(predictions)):
        print(predictions[index] , " : " , percentage_probabilities[index])
        print("-----------------------")

