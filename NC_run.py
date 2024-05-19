from plot import loss_curve_visulization
from utils import *
from MyGCN import GCNModel_for_NC
import torch.optim as optim
import torch
def train(g, model, train_mask, val_mask, num_epochs, lr, wd):
    model.train()
    best_val_acc = 0
    features, labels = g.ndata['feat'], g.ndata['label']
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    train_acc_list, val_acc_list = [], []
    train_loss_list = []
    val_loss_list = []
    for epoch in range(num_epochs):
        logits = model(g, features)  # (N, label_nums)
        pred = logits.argmax(1)
        train_loss = criterion(logits[train_mask], labels[train_mask])
        val_loss = criterion(logits[val_mask], labels[val_mask])
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        train_acc_list.append(train_acc.item())
        val_acc_list.append(val_acc.item())
        train_loss_list.append(train_loss.item())
        val_loss_list.append(val_loss.item())
        if best_val_acc < val_acc:
            best_val_acc = val_acc

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(
                'Epoch {}, train loss: {:.3f}, train acc: {:.3f}, val acc: {:.3f} (best {:.3f}) '
                .format(epoch + 1, train_loss, train_acc, val_acc, best_val_acc
                        ))   

    return train_loss_list, train_acc_list, val_loss_list, val_acc_list

def test(g, model, test_mask):
    model.eval()
    features, labels = g.ndata['feat'], g.ndata['label']
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        logits = model(g, features)  # (N, label_nums)
        pred = logits.argmax(1)
        test_loss = criterion(logits[test_mask], labels[test_mask])
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()
    return test_loss.item(), test_acc.item()

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
    dataset = Load_graph("cora")
    g = dataset[0]
    g = dgl.add_self_loop(g)
    degs = g.out_degrees().float()
    # cal D^{-1/2}
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    g.ndata['norm'] = norm.unsqueeze(1)

    train_mask, val_mask, test_mask = g.ndata['train_mask'], g.ndata[
        'val_mask'], g.ndata['test_mask']
    activation1 = F.relu
    activation2 = F.elu
    activation3 = F.leaky_relu
    epochs = 200
    hidden_size = 32
    lr = 0.01
    wd = 5e-4
    model = GCNModel_for_NC(in_feats=g.ndata['feat'].shape[1],
                        h_feats=hidden_size, num_classes= dataset.num_classes,
                        num_layers=2, bias= True, pair_norm= False, activation=activation1, dropedge=False, p=0.1)
    train_loss, train_acc, val_loss, val_acc = train(g, model, train_mask, 
                                                    val_mask, epochs, lr, wd)
    loss_curve_visulization(train_loss, train_acc, val_loss, val_acc)

    test_loss, test_acc = test(g, model, test_mask)
    print(f"loss on test set:{test_loss}")
    print(f"accuracy on test set:{test_acc}")