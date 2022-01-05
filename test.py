import os
import yaml
import torch
import argparse
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score
from dataset import create_dataloader




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--weights', type=str, default='./best.pt', help='weights path')
    parser.add_argument('--batch-size', type=int, default=128, help='total batch size for all GPUs')
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    curPath = os.path.dirname(os.path.realpath(__file__))
    yamlPath = os.path.join(curPath, "hyp.yaml")
    f = open(yamlPath, 'r', encoding='utf-8')
    hyp = yaml.load(f.read(), Loader=yaml.FullLoader)

    model = torch.load(opt.weights ,map_location=device)
    model.eval()
    _, test_loader = create_dataloader(batch_size=opt.batch_size, max_len=hyp['sequence_padding'])

    precision = 0
    recall = 0

    nb = len(test_loader)          #number of batches
    pbar = enumerate(test_loader)
    pbar = tqdm(pbar, total=nb)

    print('Evaluating...')
    for step,(x,y) in pbar:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            y_hat = model(x)
            # softmax = torch.nn.Softmax(dim=1)
            # y_score = softmax(y_hat)[:,1]
        
        y_ = y_hat.max(-1,keepdim=True)[1]
        precision += precision_score(y.cpu().numpy(), y_.cpu().numpy())
        recall += recall_score(y.cpu().numpy(),y_.cpu().numpy())
    
    precision /= nb
    recall /= nb
    f1 = 2 * precision * recall / (precision + recall)

    print('Presicion:{:.3f}, Recall:{:.3f}, F1-score:{:.3f}'.format(precision, recall, f1))
