from torch.utils.data import Dataset
import pandas as pd
from ast import literal_eval


class DKNDataset(Dataset):
    def __init__(self, behaviors_path, news_path, pad_words_num, num_clicked_news_a_user):
        super(Dataset, self).__init__()
        self.behaviors = pd.read_table(behaviors_path)
        self.behaviors.clicked_news.fillna('', inplace=True)
        self.news_with_entity = pd.read_table(news_path, index_col='id', converters={ 'title': literal_eval, 'entities': literal_eval })
        
        self.pad_words_num = pad_words_num
        self.num_clicked_news_a_user = num_clicked_news_a_user

    def __len__(self):
        return len(self.behaviors)

    def news2dict(self, news, df):
        return {
            "word": df.loc[news].title,
            "entity": df.loc[news].entities
        } if news in df.index else {
            "word": [0] * self.pad_words_num,
            "entity": [0] * self.pad_words_num
        }

    def __getitem__(self, idx):
        item = {}
        row = self.behaviors.iloc[idx]
        item["clicked"] = row.clicked
        item["candidate_news"] = self.news2dict(row.candidate_news, self.news_with_entity)
        item["clicked_news"] = [
            self.news2dict(x, self.news_with_entity)
            for x in row.clicked_news.split()[:self.num_clicked_news_a_user]
        ]
        padding = { "word": [0] * self.padding_word_num, "entity": [0] * self.padding_word_num }
        repeated_times = self.num_clicked_news_a_user - len(item["clicked_news"])
        assert repeated_times >= 0
        item["clicked_news"].extend([padding] * repeated_times)

        return item