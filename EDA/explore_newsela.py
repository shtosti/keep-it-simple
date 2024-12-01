
from dataset import NewselaDataset


# Define the path to the directory containing the .txt files
NEWSELA_DIR_PATH = "./../../datasets/newsela/newsela_article_corpus_2016-01-29/articles"
NEWSELA_METADATA_PATH = "./../../datasets/newsela/newsela_article_corpus_2016-01-29/articles_metadata.csv"

dataset = NewselaDataset(NEWSELA_DIR_PATH, NEWSELA_METADATA_PATH) # TODO add limit (to expedite exploration)
dataset.load_data()
dataset.apply_metrics()

df_newsela = dataset.get_data()

output_file_path = "newsela_with_analysis.csv"
dataset.save_to_csv(output_file_path)
log_path = "newsela_log.txt"
dataset.save_log(log_path)