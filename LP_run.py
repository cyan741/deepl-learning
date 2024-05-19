from utils import *
from MyGCN import GCNModel_for_LP
import torch
import itertools
from plot import loss_curve_visulization

def train(model, train_g, train_pos_g, train_neg_g, val_pos_g, val_neg_g):
    pred = DotPredictor()
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=0.01)
    model.train()
    train_loss_list = []
    val_loss_list = []
    val_acc_history = []
    train_acc_history = []
    for e in range(200):
        # forward
        h = model(train_g, train_g.ndata['feat'])
        pos_score = pred(train_pos_g, h)
        neg_score = pred(train_neg_g, h)
        val_pos_score = pred(val_pos_g, h)
        val_neg_score = pred(val_neg_g, h)        
        loss = compute_loss(pos_score, neg_score)
        val_loss = compute_loss(val_pos_score, val_neg_score)
        train_auc = test (model, train_pos_g, train_neg_g, h)
        val_auc = test (model, val_pos_g, val_neg_g, h)
        train_loss_list.append(loss.item())
        val_loss_list.append(val_loss.item())
        val_acc_history.append(val_auc)
        train_acc_history.append(train_auc)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if e % 5 == 0:
            print("Epoch {:03d}: train_Loss {:.4f},train_Auc {:.4f}, val_Loss {:.4f},Val_Auc {:.4f}".format(
                e, loss.item(), train_auc, val_loss.item(), val_auc))
    return train_loss_list, train_acc_history, val_loss_list, val_acc_history

def test (model, pos_g, neg_g, h):
    model.eval() 
    pred = DotPredictor()
    with torch.no_grad():
        pos_score = pred(pos_g, h)
        neg_score = pred(neg_g, h)
        auc = compute_auc(pos_score, neg_score)
    return auc
    
def set_random_seed(seed):
    random.seed(seed)                  
    np.random.seed(seed)               
    torch.manual_seed(seed)            
    torch.cuda.manual_seed_all(seed)    
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False     

if __name__ == '__main__':
    seed = 0
    set_random_seed(seed)
    dataset = Load_graph("citeseer")
    g = dataset[0]
    g = dgl.add_self_loop(g)
    degs = g.out_degrees().float()
    # cal D^{-1/2}
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    g.ndata['norm'] = norm.unsqueeze(1)

    hidden_dim = 56
    num_layers = 2
    activation = F.relu
    activation2 = F.elu
    activation3 = F.leaky_relu
    dropedge = False
 
    selfloop = True
    bias = True
    #print('Number of categories:', dataset.num_classes)
    train_g, train_pos_g, train_neg_g, val_pos_g, val_neg_g, test_pos_g, test_neg_g = edge_partition(g)

    model = GCNModel_for_LP(train_g.ndata['feat'].shape[1], 56,2, activation2, dropedge, p=0.1)
    train_loss, train_acc, val_loss, val_acc = train(model, train_g, 
                                                     train_pos_g, train_neg_g, val_pos_g, val_neg_g)
    loss_curve_visulization(train_loss, train_acc, val_loss, val_acc)
    h = model(train_g, train_g.ndata['feat'])
    test_auc = test (model, val_pos_g, val_neg_g, h)
    print(f"auc of the test set:{test_auc}")