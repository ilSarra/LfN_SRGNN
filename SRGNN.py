import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import numpy as np

torch.manual_seed(47)

# Function to retrieve dataset data of a specific split
def get_dataset(name, split):

    if split == 'complete':
        dataset = Planetoid(root='./data/', name=name)
        dataset[0].train_mask.fill_(False)
        dataset[0].train_mask[:dataset[0].num_nodes - 1000] = 1
        dataset[0].val_mask.fill_(False)
        dataset[0].val_mask[dataset[0].num_nodes - 1000:dataset[0].num_nodes - 500] = 1
        dataset[0].test_mask.fill_(False)
        dataset[0].test_mask[dataset[0].num_nodes - 500:] = 1
    else:
        dataset = Planetoid(root='./data/', name=name, split=split)

    dataset.transform = T.NormalizeFeatures()
    return dataset


# CRD module
class CRD(torch.nn.Module):
    def __init__(self, d_in, d_out, p):
        super(CRD, self).__init__()
        self.conv = GCNConv(d_in, d_out, cached=True)
        self.conv_2 = GCNConv(d_in, d_out, cached=True)
        self.p = p

    def reset_parameters(self):
        self.conv.reset_parameters()
        self.conv_2.reset_parameters()

    def forward(self, x, edge_index, mask=None):
        x_1 = F.relu(self.conv(x, edge_index))
        x_2 = F.relu(self.conv_2(x, edge_index))
        x_3 = F.dropout(x_1 + x_2, p=self.p, training=self.training)
        return x_3, [x_1, x_2]


# CLS module
class CLS(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super(CLS, self).__init__()
        self.conv = GCNConv(d_in, d_out, cached=True)
        self.conv_2 = GCNConv(d_in, d_out, cached=True)

    def reset_parameters(self):
        self.conv.reset_parameters()
        self.conv_2.reset_parameters()

    def forward(self, x, edge_index, mask=None):
        x_1 = self.conv(x, edge_index)
        x_2 = self.conv_2(x, edge_index)
        x_3 = F.log_softmax(x_1 + x_2, dim=1)
        return x_3, [x_1, x_2]


# Architecture used for experiments
class Net(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.crd_1 = CRD(dataset.num_features, 32, 0.5)
        self.crd_2 = CRD(dataset.num_features, 32, 0.5)
        self.cls = CLS(32, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x_1, [h_1, h_2] = self.crd_1(x, edge_index, data.train_mask)
        x_2, [h_3, h_4] = self.crd_2(x, edge_index, data.train_mask)
        x, [h_5, h_6] = self.cls(x_1 + x_2, edge_index, data.train_mask)
        return x, [h_1, h_2, h_3, h_4, x_1, x_2]


dataset_names = ["Cora", "CiteSeer", "PubMed"]
split_names = ["complete", "full", "public"]

# Execute test on each split of each dataset
for split in split_names:

    print("Split", split)

    for dataset_name in dataset_names:

        print("Dataset: ", dataset_name)

        # Test both with regularization and without
        for is_regularized in [False, True]:

            print("Regularization:", is_regularized)

            dataset = get_dataset(dataset_name, split)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            data = dataset[0].to(device)
            avg_acc = list()
            iterations = 10

            for k in range(iterations):

                # Initialize model in order to reset the model parameters
                model = Net(dataset).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)
                model.train()
                epochs = 200

                for epoch in range(epochs):

                    optimizer.zero_grad()
                    out, emb_vec = model(data)

                    if is_regularized:

                        reg_coeff = 1e-3

                        # Compute Frobenius norm of embedding matrices
                        emb_1 = torch.linalg.norm(emb_vec[0])
                        emb_2 = torch.linalg.norm(emb_vec[1])
                        emb_3 = torch.linalg.norm(emb_vec[2])
                        emb_4 = torch.linalg.norm(emb_vec[3])

                        # Embedding norms emb_5 and emb_6 can be used to consider the outputs
                        # of crd_1 and crd_2 in the regularization

                        emb_5 = torch.linalg.norm(emb_vec[4])
                        emb_6 = torch.linalg.norm(emb_vec[5])

                        # Compute the result as the abs. difference between the norms
                        res_1 = abs(emb_1 - emb_2)
                        res_2 = abs(emb_3 - emb_4)
                        res_3 = 0

                        # Uncomment to consider emb_5 and emb_6 in the regularization
                        # res_3 = abs(emb_5 - emb_6)

                        # Avoid zero division
                        eps = 1e-5

                        # Compute regularization
                        reg_norm = 1 / (res_1 + res_2 + res_3 + eps)

                        nll = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
                        total_reg = reg_coeff * reg_norm

                        loss = nll + total_reg

                    else:
                        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

                    loss.backward()
                    optimizer.step()

                model.eval()
                pred, embedding = model(data)
                pred = pred.argmax(dim=1)
                correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
                acc = int(correct) / int(data.test_mask.sum())
                avg_acc.append(acc)

            print("Average Accuracy: %.4f +/- %.4f" % (np.mean(avg_acc) * 100, np.std(avg_acc) * 100))
        print("\n")
    print("\n")
