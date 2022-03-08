import argparse
import os
import json
from tqdm import tqdm

import utils.utils as utils

import numpy as np
import pandas as pd


def clean_behaviors(source, target):
    print(f"Clean up {source}")
    behaviors = pd.read_table(source, header=None, usecols=[2, 3], names=['clicked_news', 'impressions'])
    behaviors.impressions = behaviors.impressions.str.split()
    behaviors = behaviors.explode('impressions').reset_index(drop=True)
    split = behaviors.impressions.str.split('-', expand = True)
    behaviors = behaviors.assign(candidate_news=split[0], clicked=split[1])
    behaviors['clicked_news'].fillna('', inplace=True)
    behaviors.to_csv(target, sep='\t', index=False, columns=['clicked_news', 'candidate_news', 'clicked'])

def balance(source, target, true_false_division_range):
    low = true_false_division_range[0]
    high = true_false_division_range[1]
    assert low <= high
    original = pd.read_table(source)
    true_part = original[original['clicked'] == 1]
    false_part = original[original['clicked'] == 0]

    if len(true_part) / len(false_part) < low:
        print(f'Drop {len(false_part) - int(len(true_part) / low)} from false part')
        false_part = false_part.sample(n=int(len(true_part) / low))
    elif len(true_part) / len(false_part) > high:
        print(f'Drop {len(true_part) - int(len(false_part) * high)} from true part')
        true_part = true_part.sample(n=int(len(false_part) * high))

    balanced = pd.concat([true_part, false_part]).sample(frac=1).reset_index(drop=True)
    balanced.to_csv(target, sep='\t', index=False)

    
def clean_news(source, target, word2idx_path, entity2idx_path, mode, word_freq_threshold, entity_freq_threshold, entity_confidence_threshold, pad_words_num):
    if mode == 'train':
        word2idx = {}
        entity2idx = {}
        word2freq = {}
        entity2freq = {}

        news = pd.read_table(source, header=None, usecols=[0, 3, 6], names=['id', 'title', 'entities'])
        news.entities.fillna('[]', inplace=True)
        parsed_news = pd.DataFrame(columns=['id', 'title', 'entities'])

        with tqdm(total=len(news), desc="Counting token and entities") as t:
            for row in news.itertuples(index=False):
                for w in utils.clean_text(row.title).split():
                    if w not in word2freq:
                        word2freq[w] = 1
                    else:
                        word2freq[w] += 1
                for e in json.loads(row.entities):
                    times = len(list(filter(lambda x: x < len(row.title), e['OccurrenceOffsets']))) * e['Confidence']
                    if times > 0:
                        if e['WikidataId'] not in entity2freq:
                            entity2freq[e['WikidataId']] = times
                        else:
                            entity2freq[e['WikidataId']] += times
                t.update(1)

        for k, v in word2freq.items():
            if v >= word_freq_threshold:
                word2idx[k] = len(word2idx) + 1

        for k, v in entity2freq.items():
            if v >= entity_freq_threshold:
                entity2idx[k] = len(entity2idx) + 1

        with tqdm(total=len(news), desc="Parsing words and entities") as t:
            for row in news.itertuples(index=False):
                new_row = [row.id, [0] * pad_words_num, [0] * pad_words_num ]
                local_entity_map = {}
                for e in json.loads(row.entities):
                    if e['Confidence'] > entity_confidence_threshold and e['WikidataId'] in entity2idx:
                        for x in ' '.join(e['SurfaceForms']).lower().split():
                            local_entity_map[x] = entity2idx[e['WikidataId']]
                try:
                    for i, w in enumerate(utils.clean_text(row.title).split()):
                        if w in word2idx:
                            new_row[1][i] = word2idx[w]
                            if w in local_entity_map:
                                new_row[2][i] = local_entity_map[w]
                except IndexError:
                    pass

                parsed_news.loc[len(parsed_news)] = new_row
                t.update(1)

        parsed_news.to_csv(target, sep='\t', index=False)
        pd.DataFrame(word2idx.items(), columns=['word','int']).to_csv(word2idx_path, sep='\t', index=False)
        pd.DataFrame(entity2idx.items(), columns=['entity', 'int']).to_csv(entity2idx_path, sep='\t', index=False)
    
    elif mode == 'test':
        news = pd.read_table(source, header=None, usecols=[0, 3, 6], names=['id', 'title', 'entities'])
        news.entities.fillna('[]', inplace=True)
        parsed_news = pd.DataFrame(columns=['id', 'title', 'entities'])

        word2idx = dict(pd.read_table(word2idx_path).values.tolist())
        entity2idx = dict(pd.read_table(entity2idx_path).values.tolist())

        word_total = 0
        word_missed = 0

        with tqdm(total=len(news), desc="Parsing words and entities") as t:
            for row in news.itertuples(index=False):
                new_row = [ row.id, [0] * pad_words_num, [0] * pad_words_num]
                local_entity_map = {}

                for e in json.loads(row.entities):
                    if e['Confidence'] > entity_confidence_threshold and e['WikidataId'] in entity2idx:
                        for x in ' '.join(e['SurfaceForms']).lower().split():
                            local_entity_map[x] = entity2idx[e['WikidataId']]
                try:
                    for i, w in enumerate(utils.clean_text(row.title).split()):
                        word_total += 1
                        if w in word2idx:
                            new_row[1][i] = word2idx[w]
                            if w in local_entity_map:
                                new_row[2][i] = local_entity_map[w]
                        else:
                            word_missed += 1
                except IndexError:
                    pass
                parsed_news.loc[len(parsed_news)] = new_row
                t.update(1)

        parsed_news.to_csv(target, sep='\t', index=False)
        print(f'Out-of-Vocabulary rate: {word_missed/word_total:.4f}')
    else:
        print('Wrong mode!')

def transform_entity_embedding(source, target, entity2idx_path, kb_dim):
    entity_embedding = pd.read_table(source, header=None)
    entity_embedding['vector'] = entity_embedding.iloc[:, 1:101].values.tolist()
    entity_embedding = entity_embedding[[0, 'vector']].rename(columns={0: "entity"})
    
    entity2int = pd.read_table(entity2idx_path)
    merged_df = pd.merge(entity_embedding, entity2int, on='entity').sort_values('int')
    entity_embedding_transformed = np.zeros((len(entity2int) + 1, kb_dim))
    
    for row in merged_df.itertuples(index=False):
        entity_embedding_transformed[row.int] = row.vector

    np.save(target, entity_embedding_transformed)

parser = argparse.ArgumentParser(description='Preprocess DKN PyTorch')
parser.add_argument("--dataset_path", default="./data", help="Path to dataset.")
parser.add_argument("--model_dir", default="./experiments/base_model", help="Path to model checkpoint (by default train from scratch).")

if __name__ == '__main__':
    args = parser.parse_args()
    
    # path setting
    train_behaviors_path = os.path.join(args.dataset_path, 'train/behaviors.tsv')
    train_behaviors_o_path = os.path.join(args.dataset_path, 'train/behaviors_cleaned.tsv')
    train_news_path = os.path.join(args.dataset_path, 'train/news.tsv')
    train_news_o_path = os.path.join(args.dataset_path, 'train/news_cleaned.tsv')

    test_behaviors_path = os.path.join(args.dataset_path, 'test/behaviors.tsv')
    test_behaviors_o_path = os.path.join(args.dataset_path, 'test/behaviors_cleaned.tsv')
    test_news_path = os.path.join(args.dataset_path, 'test/news.tsv')
    test_news_o_path = os.path.join(args.dataset_path, 'test/news_cleaned.tsv')
    
    entity_emb_path = os.path.join(args.dataset_path, 'entity_embedding.vec')
    entity_emb_o_path = os.path.join(args.dataset_path, 'entity_embedding.npy')
    word2idx_path = os.path.join(args.dataset_path, 'word2idx.tsv')
    entity2idx_path = os.path.join(args.dataset_path, 'entity2idx.tsv')
    params_path = os.path.join(args.model_dir, 'params.json')
    params = utils.Params(params_path)

    # main 
    print('Process data for training')
    print('Clean up behaviors data')
    clean_behaviors(train_behaviors_path, train_behaviors_o_path)

    #print('Balance behaviors data')
    #balance(path.join(train_dir, 'behaviors_cleaned.tsv'), path.join(train_dir, 'behaviors_cleaned_balanced.tsv'), (1 / 2, 2))

    print('Clean up news data')
    
    clean_news(train_news_path, train_news_o_path, word2idx_path, entity2idx_path, mode='train', 
                word_freq_threshold=params.word_freq_threshold, entity_freq_threshold=params.entity_freq_threshold, entity_confidence_threshold=params.entity_confidence_threshold, pad_words_num=params.pad_words_num)

    print('Transform entity embeddings')
    transform_entity_embedding(entity_emb_path, entity_emb_o_path, entity2idx_path, params.kb_dim)

    print('\nProcess data for evaluation')
    print('Clean up behaviors data')
    clean_behaviors(test_behaviors_path, test_behaviors_o_path)

    print('Clean up news data')
    clean_news(test_news_path, test_news_o_path, word2idx_path, entity2idx_path, mode='test', 
               word_freq_threshold=params.word_freq_threshold, entity_freq_threshold=params.entity_freq_threshold, entity_confidence_threshold=params.entity_confidence_threshold, pad_words_num=params.pad_words_num))