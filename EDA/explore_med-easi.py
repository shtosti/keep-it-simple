from dataset import MedEASiDataset

dataset = MedEASiDataset(file_path="./../../datasets/Med-EASi/Med-EASi.full.ori.csv")
dataset.load_data()
dataset.apply_metrics()
dataset.save_to_csv("Med-EASi_with_analysis.csv")
dataset.save_log("Med-EASi_log.txt")