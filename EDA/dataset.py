import os
import pandas as pd
import spacy
import textstat
from lexicalrichness import LexicalRichness


class TSDataset:
    def __init__(self):
        self.data_df = pd.DataFrame()
        self.nlp_EN = spacy.load("en_core_web_sm")  # English model
        self.nlp_ES = spacy.load("es_core_news_sm")  # Spanish model (used if needed)

    def word_count_spacy(self, text, nlp=None):
        nlp = nlp or self.nlp_EN  # Default to English NLP model
        doc = nlp(text)
        return len([token.text for token in doc])

    def sentence_count_spacy(self, text, nlp=None):
        nlp = nlp or self.nlp_EN
        doc = nlp(text)
        return len(list(doc.sents))

    def character_count(self, text):
        return len(text)

    def measure_flesh_reading_ease(self, text):
        return textstat.flesch_reading_ease(text)

    def measure_difficult_words(self, text):
        return textstat.difficult_words(text)

    def measure_type_token_ratio(self, text):
        lex = LexicalRichness(text)
        if lex.words == 0:
            return 0
        return lex.ttr

    def apply_metrics(self, column_prefix=""):
        """Apply metrics to the dataset."""
        for col in ["source", "target"]:
            col_name = f"{column_prefix}{col}" if column_prefix else col
            if col_name in self.data_df.columns:
                self.data_df[f"num_tokens_{col}"] = self.data_df[col_name].apply(self.word_count_spacy)
                self.data_df[f"num_sentences_{col}"] = self.data_df[col_name].apply(self.sentence_count_spacy)
                self.data_df[f"num_characters_{col}"] = self.data_df[col_name].apply(self.character_count)
                self.data_df[f"flesh_reading_ease_{col}"] = self.data_df[col_name].apply(self.measure_flesh_reading_ease)
                self.data_df[f"difficult_words_{col}"] = self.data_df[col_name].apply(self.measure_difficult_words)
                self.data_df[f"type_token_ratio_{col}"] = self.data_df[col_name].apply(self.measure_type_token_ratio)

    def save_to_csv(self, file_path):
        self.data_df.to_csv(file_path, index=False)
        print(f"Data successfully saved to {file_path}")

    def save_log(self, log_file_path):
        with open(log_file_path, "w") as log_file:
            log_file.write("DataFrame Info:\n")
            self.data_df.info(buf=log_file)
            log_file.write("\n\nDataFrame Describe:\n")
            self.data_df.describe(include="all").to_string(buf=log_file)
            print(f"Log saved to {log_file_path}")


class NewselaDataset(TSDataset):
    def __init__(self, data_dir, metadata_path, limit=None):
        super().__init__()
        self.data_dir = data_dir
        self.metadata_path = metadata_path
        self.limit = limit

    def load_data(self):
        metadata = pd.read_csv(self.metadata_path, sep=",")
        data = []
        loaded_files = 0

        for filename in os.listdir(self.data_dir):
            if filename.endswith(".txt"):
                file_path = os.path.join(self.data_dir, filename)
                with open(file_path, "r", encoding="utf-8") as file:
                    text = file.read()
                    data.append({"filename": filename, "content": text})
                    loaded_files += 1
                    if self.limit and loaded_files >= self.limit:
                        break

        articles_df = pd.DataFrame(data)
        self.data_df = pd.merge(articles_df, metadata, on="filename", how="inner")


class WikiLargeDataset(TSDataset):
    def __init__(self, base_path, base_filename):
        super().__init__()
        self.base_path = base_path
        self.base_filename = base_filename

    def load_data(self):
        valid_src_path = os.path.join(self.base_path, f"{self.base_filename}valid.src")
        valid_dst_path = os.path.join(self.base_path, f"{self.base_filename}valid.dst")

        with open(valid_src_path, "r") as src_file, open(valid_dst_path, "r") as dst_file:
            valid_src = src_file.readlines()
            valid_dst = dst_file.readlines()

        valid_df = pd.DataFrame({"source": valid_src, "target": valid_dst})
        self.data_df = valid_df[valid_df["source"].str.strip().astype(bool) & valid_df["target"].str.strip().astype(bool)]


class MedEASiDataset(TSDataset):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def load_data(self):
        self.data_df = pd.read_csv(self.file_path)
        self.data_df.rename(columns={"Expert": "source", "Simple": "target"}, inplace=True)
        self.data_df = self.data_df[
            self.data_df["source"].str.strip().astype(bool) & self.data_df["target"].str.strip().astype(bool)
        ]
