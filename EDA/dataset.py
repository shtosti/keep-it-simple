import os
import pandas as pd
import spacy
import textstat
from lexicalrichness import LexicalRichness

# TODO create a super class and refactor child classes

class NewselaDataset:
    def __init__(self, data_dir, metadata_path, limit=None):
        self.data_dir = data_dir
        self.metadata_path = metadata_path
        self.nlp_EN = spacy.load("en_core_web_sm")  # English model
        self.nlp_ES = spacy.load("es_core_news_sm")  # Spanish model
        self.articles_df = pd.DataFrame()
        self.merged_df = pd.DataFrame()
        self.limit = limit

    def load_data(self):
        metadata = pd.read_csv(self.metadata_path, sep=',')

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


        self.articles_df = pd.DataFrame(data)
        self.merged_df = pd.merge(self.articles_df, metadata, on="filename", how="inner")

    def word_count_spacy(self, text, nlp):
        doc = nlp(text)
        return len([token.text for token in doc])

    def sentence_count_spacy(self, text, nlp):
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
        return lex.ttr 

    def apply_metrics(self):

        def select_nlp(text, language):
            # Select the correct language model based on the lang column
            if language == "es":
                return self.nlp_ES
            else:
                return self.nlp_EN

        self.merged_df["num_tokens"] = self.merged_df.apply(
            lambda row: self.word_count_spacy(row["content"], select_nlp(row["content"], row["language"])), axis=1
        )
        self.merged_df["num_sentences"] = self.merged_df.apply(
            lambda row: self.sentence_count_spacy(row["content"], select_nlp(row["content"], row["language"])), axis=1
        )
        self.merged_df["num_characters"] = self.merged_df["content"].apply(self.character_count)
        self.merged_df["flesh_reading_ease"] = self.merged_df["content"].apply(self.measure_flesh_reading_ease)
        self.merged_df["difficult_words"] = self.merged_df["content"].apply(self.measure_difficult_words)
        self.merged_df["type-token_ratio"] = self.merged_df["content"].apply(self.measure_type_token_ratio)

    def get_data(self):
        return self.merged_df
    
    def save_to_csv(self, file_path):
        self.merged_df.to_csv(file_path, index=False)
        print(f"Data successfully saved to {file_path}")

    def save_log(self, log_path_path):
        with open(log_path_path, "w") as log_file:
            log_file.write("DataFrame Info:\n")
            self.merged_df.info(buf=log_file)
            log_file.write("\n\nDataFrame Describe:\n")
            log_file.write(str(self.merged_df.describe()))
            log_file.write("\n\nLanguages:\n")
            log_file.write(str(self.merged_df["language"].value_counts()))
            log_file.write("\n\nGrade Levels:\n")
            log_file.write(str(self.merged_df["grade_level"].value_counts()))
        print(f"Log saved to {log_path_path}")


class WikiLargeDataset:
    def __init__(self, base_path, base_filename):
        self.base_path = base_path
        self.base_filename = base_filename
        self.nlp = spacy.load("en_core_web_sm")  # English model
        self.data_df = pd.DataFrame()

    def load_data(self):
        # File paths
        train_src_path = f"{self.base_path}/{self.base_filename}train.src"
        train_dst_path = f"{self.base_path}/{self.base_filename}train.dst"
        test_src_path = f"{self.base_path}/{self.base_filename}test.src"
        test_dst_path = f"{self.base_path}/{self.base_filename}test.dst"
        valid_src_path = f"{self.base_path}/{self.base_filename}valid.src"
        valid_dst_path = f"{self.base_path}/{self.base_filename}valid.dst"

        # Read data
        with open(train_src_path, "r") as src, open(train_dst_path, "r") as dst:
            train_src = src.readlines()
            train_dst = dst.readlines()
        with open(test_src_path, "r") as src, open(test_dst_path, "r") as dst:
            test_src = src.readlines()
            test_dst = dst.readlines()
        with open(valid_src_path, "r") as src, open(valid_dst_path, "r") as dst:
            valid_src = src.readlines()
            valid_dst = dst.readlines()

        # Create DataFrames for each split
        train_df = pd.DataFrame({"source": train_src, "target": train_dst, "split": "train"})
        test_df = pd.DataFrame({"source": test_src, "target": test_dst, "split": "test"})
        valid_df = pd.DataFrame({"source": valid_src, "target": valid_dst, "split": "valid"})

        # Concatenate all DataFrames
        self.data_df = pd.concat([train_df, test_df, valid_df], ignore_index=True)

    def word_count_spacy(self, text):
        doc = self.nlp(text)
        return len([token.text for token in doc])

    def sentence_count_spacy(self, text):
        doc = self.nlp(text)
        return len(list(doc.sents))

    def character_count(self, text):
        return len(text)

    def measure_flesh_reading_ease(self, text):
        return textstat.flesch_reading_ease(text)

    def measure_difficult_words(self, text):
        return textstat.difficult_words(text)

    def measure_type_token_ratio(self, text):
        lex = LexicalRichness(text)
        return lex.ttr 

    def apply_metrics(self):
        self.data_df["num_tokens_source"] = self.data_df["source"].apply(self.word_count_spacy)
        self.data_df["num_sentences_source"] = self.data_df["source"].apply(self.sentence_count_spacy)
        self.data_df["num_characters_source"] = self.data_df["source"].apply(self.character_count)
        self.data_df["flesh_reading_ease_source"] = self.data_df["source"].apply(self.measure_flesh_reading_ease)
        self.data_df["difficult_words_source"] = self.data_df["source"].apply(self.measure_difficult_words)
        self.data_df["type_token_ratio_source"] = self.data_df["source"].apply(self.measure_type_token_ratio)

        self.data_df["num_tokens_target"] = self.data_df["target"].apply(self.word_count_spacy)
        self.data_df["num_sentences_target"] = self.data_df["target"].apply(self.sentence_count_spacy)
        self.data_df["num_characters_target"] = self.data_df["target"].apply(self.character_count)
        self.data_df["flesh_reading_ease_target"] = self.data_df["target"].apply(self.measure_flesh_reading_ease)
        self.data_df["difficult_words_target"] = self.data_df["target"].apply(self.measure_difficult_words)
        self.data_df["type_token_ratio_target"] = self.data_df["target"].apply(self.measure_type_token_ratio)

    def save_to_csv(self, file_path):
        self.data_df.to_csv(file_path, index=False)
        print(f"Data successfully saved to {file_path}")

    def save_log(self, log_file_path):
        with open(log_file_path, "w") as log_file:
            log_file.write("DataFrame Info:\n")
            self.data_df.info(buf=log_file)
            log_file.write("\n\nDataFrame Describe:\n")
            log_file.write(str(self.data_df.describe()))
            log_file.write("\n\nSplit Counts:\n")
            log_file.write(str(self.data_df["split"].value_counts()))
        print(f"Log saved to {log_file_path}")
