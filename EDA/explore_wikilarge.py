from dataset import WikiLargeDataset


BASE_PATH = "./../../datasets/wiki/wikilarge"
BASE_FILENAME = "wiki.full.aner.ori."

wikilarge_dataset = WikiLargeDataset(BASE_PATH, BASE_FILENAME)
wikilarge_dataset.load_data()
wikilarge_dataset.apply_metrics()
wikilarge_dataset.save_to_csv(f"{BASE_PATH}/wikilarge_with_analysis.csv")
wikilarge_dataset.save_log(f"{BASE_PATH}/wikilarge_log.txt")
