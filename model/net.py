import torch
import torch.nn as nn
from model.kcnn import KCNN
from model.attention import Attention


class Net(torch.nn.Module):
    def __init__(self, window_sizes, num_filters, pos_weight,  
                    entity_embedding, context_embedding, device):
        super(Net, self).__init__()
        self.kcnn = KCNN(config, entity_embedding, context_embedding)
        self.attention = Attention(config)
        
        self.dnn = nn.Sequential(nn.Linear(len(window_sizes) * 2 * num_filters, 16), nn.Linear(16, 1))
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).float().to(device))

    def forward(self, candidate_news, clicked_news):
        """
        Args:
          candidate_news:
            {
                "word": [Tensor(batch_size) * num_words_a_news],
                "entity":[Tensor(batch_size) * num_words_a_news]
            }
          clicked_news:
            [
                {
                    "word": [Tensor(batch_size) * num_words_a_news],
                    "entity":[Tensor(batch_size) * num_words_a_news]
                } * num_clicked_news_a_user
            ]
        Returns:
          click_probability: batch_size
        """
        # batch_size, len(window_sizes) * num_filters
        candidate_news_vector = self.kcnn(candidate_news)
        # num_clicked_news_a_user, batch_size, len(window_sizes) * num_filters
        clicked_news_vector = torch.stack([self.kcnn(x) for x in clicked_news])
        # batch_size, len(window_sizes) * num_filters
        user_vector = self.attention(candidate_news_vector, clicked_news_vector)
        
        # Sigmoid is done with BCEWithLogitsLoss
        # batch_size
        click_probability = self.dnn(torch.cat((user_vector, candidate_news_vector),
                                               dim=1)).squeeze(dim=1)
        return click_probability