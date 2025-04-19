from sklearn.metrics import accuracy_score
from copy import deepcopy
import os
import tqdm
import torch
import torch.nn.functional as F

from random import random

device = "cpu" # not enough memory

class ModelTraining:
    def __init__(self, model, data):
        
        # training
        self.training_logs = {}
        self.training_logs["iter"] = []
        self.training_logs["epoch"] = []
        self.training_logs["training_loss"] = []
        self.training_logs["training_acc"] = []
        self.training_logs["val_loss"] = []       
        self.training_logs["val_acc"] = []
        self.training_logs["unlabelled_count"] = []
        # model
        self.model = model.to(device)
        
        self.data = data

        self.optimizer = torch.optim.Adam(self.model.parameters(),lr = 0.001)

        self.t_means = (self.data["X"][self.data["train_mask"]]).mean(dim=0, keepdim=True)
        self.t_stds = (self.data["X"][self.data["train_mask"]]).std(dim=0, keepdim=True)
        
    
    def normalize(self, data):

        normalized_data = (data - self.t_means) / (self.t_stds+1e-10)
        return normalized_data

    def train_model(self):
        

        torch.manual_seed(47)
            
        self.model.train()
        # Sets the gradients of all optimized tensors to zero
        self.optimizer.zero_grad()
        
        X_data = self.normalize(self.data["X"]).to(device)

        predictions = self.model(X_data, self.data["edge_index"].to(device))[self.data["train_mask"]]
        ground_truth = (self.data["y"]).type(torch.float)[self.data["train_mask"]]
        # Compute loss (here CrossEntropyLoss)
        loss = torch.nn.BCEWithLogitsLoss()(torch.reshape(predictions,(-1,)), ground_truth).to(device)
        # BackProp + Gradient Descent
        (loss).backward()
        self.optimizer.step()
        
        # metrics
        self.training_logs["training_loss"].append(loss.item())
        train_accuracy = accuracy_score(ground_truth.cpu().numpy(), (predictions>0.5).type(torch.long).cpu().numpy() )
        self.training_logs["training_acc"].append(train_accuracy)
        
        
            
    def val_model(self):
        self.model.eval()
        with torch.inference_mode():
            predictions = self.model(self.normalize(self.data["X"]).to(device), self.data["edge_index"].to(device))[self.data["val_mask"]]
            ground_truth = (self.data["y"]).type(torch.float)[self.data["val_mask"]]
            loss = torch.nn.BCEWithLogitsLoss()(torch.reshape(predictions,(-1,)), ground_truth).to(device)
        
        # metrics
        self.training_logs["val_loss"].append(loss.item())
        val_accuracy = accuracy_score(ground_truth.cpu().numpy(), (predictions>0.5).type(torch.long).cpu().numpy() )
        self.training_logs ["val_acc"].append(val_accuracy)


    def update_training_and_unlabelled_mask(self, random_threshold=0.8):

        # need to change to Crosee Entropy loss for easier stuff
        original_unlabelled = (self.data.unlabelled_mask == True).nonzero(as_tuple=False)

        # make predictions

        self.model.eval()
        with torch.inference_mode():
            predictions = self.model(self.normalize(self.data["X"]).to(device), self.data["edge_index"].to(device))
            prob = F.sigmoid(predictions)
        for i in original_unlabelled:
            r = random()
            if r>random_threshold:
                pass
            else:
                continue                

            idx = int(i)
            if prob[idx] >0.98:
                self.data["y"][idx] = 1
                self.data["train_mask"][idx] = True
                self.data["unlabelled_mask"][idx] = False
            elif (prob[idx]<0.02) and (prob[idx]>0):
                self.data["y"][idx] = 0
                self.data["train_mask"][idx] = True
                self.data["unlabelled_mask"][idx] = False

    def regular_training_loop(self, model_path, n_epochs=101):
        best_loss = float("inf")

        self.weight_init(model_path)

        with tqdm.tqdm(range(1,n_epochs), unit = "epoch") as tepoch:
            for epoch in tepoch:

                self.training_logs["epoch"].append(epoch)
                self.training_logs["iter"].append(0)
                self.training_logs["unlabelled_count"].append(torch.sum(self.data["unlabelled_mask"]))

                self.train_model()
                self.val_model()
                if self.training_logs["val_loss"][-1]<best_loss:
                    torch.save(self.model.state_dict(), model_path/"best_ind_model.pth")
                    best_loss = self.training_logs["val_loss"][-1]
            
                tepoch.set_postfix(training_loss=self.training_logs["training_loss"][-1], 
                                validation_loss=self.training_logs["val_loss"][-1], 
                                training_accuracy=100. * self.training_logs["training_acc"][-1],
                                validation_accuracy=100. * self.training_logs["val_acc"][-1],
                                )

                    
                    
    def self_supervised_training_loop(self, model_path, n_epochs=101, n_iters=5, threshold = 0.8):
        best_loss = float("inf")
        
        self.weight_init(model_path)
        with tqdm.tqdm(range(1,n_iters), unit = "Iteration", position=1, leave=False) as t_iter:
            
            for iter in t_iter:
                with tqdm.tqdm(range(1,n_epochs), unit = "epoch", position=0, leave=False) as tepoch:
                    for epoch in tepoch:

                        self.training_logs["epoch"].append(epoch)
                        self.training_logs["iter"].append(iter)
                        self.training_logs["unlabelled_count"].append(torch.sum(self.data["unlabelled_mask"]))

                        self.train_model()
                        self.val_model()
                        if self.training_logs["val_loss"][-1]<best_loss:
                            torch.save(self.model.state_dict(), model_path/"best_semi_sup_model.pth")
                            best_loss = self.training_logs["val_loss"][-1]
                    
                        tepoch.set_postfix(training_loss=self.training_logs["training_loss"][-1], 
                                        validation_loss=self.training_logs["val_loss"][-1], 
                                        training_accuracy=100. * self.training_logs["training_acc"][-1],
                                        validation_accuracy=100. * self.training_logs["val_acc"][-1],
                                        )

                    self.update_training_and_unlabelled_mask(threshold)

                    t_iter.set_postfix(training_loss=self.training_logs["training_loss"][-1], 
                                        validation_loss=self.training_logs["val_loss"][-1], 
                                        training_accuracy=100. * self.training_logs["training_acc"][-1],
                                        validation_accuracy=100. * self.training_logs["val_acc"][-1],
                                        unlabelled_count = self.training_logs["unlabelled_count"][-1]
                                        )


            
    
    def weight_init(self, model_path):
        if not os.path.exists(model_path/"random_wts.pth"):
            torch.save(self.model.state_dict(), model_path/"random_wts.pth")
        
        self.model.load_state_dict(torch.load(model_path/"random_wts.pth", weights_only=True))
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr = 0.0001)