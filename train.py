import argparse
import os
import logging
import numpy as np
from tqdm import tqdm

import model.net as net
import model.data_loader as data_loader
from evaluate import evaluate
import utils.utils as utils

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim


parser = argparse.ArgumentParser(description='DKN PyTorch')
parser.add_argument("--dataset_path", default="./data", help="Path to dataset.")
parser.add_argument("--model_dir", default="./experiments/base_model", help="Path to model checkpoint (by default train from scratch).")

def main():
    args = parser.parse_args()

    # torch setting
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # path setting
    path = args.dataset_path
    train_behaviors_dataset_path = os.path.join(path, "data/train/behaviors_cleaned.tsv")
    train_news_dataset_path = os.path.join(path, "data/train/news_cleaned.tsv")
    checkpoint_dir = os.path.join(args.model_dir, "checkpoint")
    params_path = os.path.join(args.model_dir, 'params.json')
    
    # load npy file
    entity_embedding = np.load('data/train/entity_embedding.npy')
    context_embedding = np.load('data/train/entity_embedding.npy')
    
    # params
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))
    params = utils.Params(params_path)
    params.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # data loader
    logging.info("Process DataLoader...")
    dataset = data_loader.DKNDataset(train_behaviors_dataset_path, train_news_dataset_path
                                pad_words_num= params.pad_words_num, num_clicked_news_a_user=params.num_clicked_news_a_user)
    
    # split train & valid dataset
    train_size = int(params.train_validation_split[0] / sum(params.train_validation_split) * len(dataset))
    validation_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, (train_size, validation_size))
    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)

    # net params
    model = net.Net(entity_embedding= entity_embedding, context_embedding=context_embedding, 
                    device=params.device)
    model.to(params.device)
    ddp_model = DDP(model, device_ids=[0,1,2,3,4,5,6,7])
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=params.learning_rate)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([23.7]).float().to(params.device))
    optimizer.zero_grad()

    best_score = -float("inf")
    no_update = 0
    logging.info('----------------------------------------------')
    print(ddp_model)
    
    logging.info("Starting training...")
    for epoch_id in range(0, params.epochs):
        # train mod
        print("Epoch {}/{}".format(epoch_id, params.epochs))
        ddp_model.train()
        running_loss = 0
      
        with tqdm(total=len(train_generator)) as t:
            for i_batch, minibatch in enumerate(train_generator):
                ddp_model.zero_grad()
                y_pred = ddp_model(minibatch["candidate_news"], minibatch["clicked_news"])
                y = minibatch["clicked"].float().to(params.device)
                loss = criterion(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                t.set_postfix(loss=running_loss/((i_batch+1)*params.batch_size))
                t.update()
        
        if epoch_id % params.valid_every == 0:
            ddp_model.eval()
            print('valid')
            # validation_generator = data_loader.data_generator(valid_data)

            # avg_recall, avg_precision, avg_f1  = evaluate(model=model, data_generator=validation_generator, device=params.device, margin=params.margin)
            
            #logging.info("- Eval: Epoch: {}: metrics f1: {}, recall: {}, precision: {}".format(epoch_id, avg_f1, avg_recall, avg_precision))
            '''
            if avg_f1 > best_score:
                best_score = avg_f1
                utils.save_checkpoint(checkpoint_dir, model, optimizer, epoch_id, best_score)
            '''

           
if __name__=='__main__':
    main()