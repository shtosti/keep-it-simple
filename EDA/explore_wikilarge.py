from dataset import WikiLargeDataset


BASE_PATH = "./../../datasets/wiki/wikilarge"
BASE_FILENAME = "wiki.full.aner.ori."

wikilarge_dataset = WikiLargeDataset(BASE_PATH, BASE_FILENAME)
wikilarge_dataset.load_data()
wikilarge_dataset.apply_metrics()
wikilarge_dataset.save_to_csv(f"wikilarge_with_analysis_valid.csv")
wikilarge_dataset.save_log(f"wikilarge_log_valid.txt")
