from sklearn.metrics import accuracy_score
from copy import deepcopy
import os
import tqdm
import torch
import torch.nn.functional as F

from torch_geometric.loader import NeighborLoader
from random import random

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


class ModelTraining:
    """
        Class Representing methods to implement the Graph Neural Network Training loop.
        
        Attributes
        ----------
            - training_logs: dictionary used for logging training behavior.
            - model: Model used
            - data: Graph Dataset (should be fully processed)
            - optimizer: optimizer used (Adam used for project)
            - t_means, t_stds: mean and standard deviation of training set. Used for data normalization.
        
        Methods
        -------
            - init__: Constructor of the class.
            - normalize: Helper method to return normalized data.
            - __train_model: Training portion of the training loop.
            - __val_model: Validation portion of the training loop.
            - update_training_and_unlabelled_mask: Logic for updating training and unlabelled masks.
            - regular_training_loop: Regular training loop.
            - self_supervised_training_loop: Semi-supervised training loop.
            - weight_init: Method for re-inintializing a model.
            - predictions: Method for retrieving model predictions based on training mask.
        """
    def __init__(self, model, data):
        """
            Constructor of the class.

            Parameters
            ----------
                model: GNN_Model
                data: dataset        
        """
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
        
        self.train_loader = NeighborLoader(self.data, num_neighbors=[10]*2, input_nodes=self.data.train_mask, batch_size=64)
    
    def normalize(self, data):
        """
            Helper method to return normalized data.
            
            Parameters
            ----------
                data: training/validation/test data
            
            Returns
            -------
                data (standard scaled) 
        """

        normalized_data = (data - self.t_means) / (self.t_stds+1e-10)
        return normalized_data

    def __train_model(self):
        """
            Training portion of the training loop.
        """

        torch.manual_seed(47)
            
        self.model.train()
        # Sets the gradients of all optimized tensors to zero
        self.optimizer.zero_grad()
        
        X_data = self.normalize(self.data["X"]).to(device)

        predictions = self.model(X_data, self.data["edge_index"].to(device))[self.data["train_mask"]]
        ground_truth = (self.data["y"]).type(torch.float).to(device)[self.data["train_mask"]]
        # Compute loss (here CrossEntropyLoss)
        loss = torch.nn.BCEWithLogitsLoss()(torch.reshape(predictions,(-1,)), ground_truth).to(device)
        # BackProp + Gradient Descent
        (loss).backward()
        self.optimizer.step()
        
        # metrics
        self.training_logs["training_loss"].append(loss.item())
        train_accuracy = accuracy_score(ground_truth.cpu().numpy(), (predictions>0.5).type(torch.long).cpu().numpy() )
        self.training_logs["training_acc"].append(train_accuracy)
        
        
            
    def __val_model(self):
        """
            Validation portion of the training loop.
        """

        self.model.eval()
        with torch.inference_mode():
            predictions = self.model(self.normalize(self.data["X"]).to(device), self.data["edge_index"].to(device))[self.data["val_mask"]]
            ground_truth = (self.data["y"]).type(torch.float).to(device)[self.data["val_mask"]]
            loss = torch.nn.BCEWithLogitsLoss()(torch.reshape(predictions,(-1,)), ground_truth).to(device)
        
        # metrics
        self.training_logs["val_loss"].append(loss.item())
        val_accuracy = accuracy_score(ground_truth.cpu().numpy(), (predictions>0.5).type(torch.long).cpu().numpy() )
        self.training_logs ["val_acc"].append(val_accuracy)


    def update_training_and_unlabelled_mask(self, random_threshold=0.8):
        """
            Logic for updating training and unlabelled masks.

            Parameters
            ----------
                random_threshold: probility for ignoring to label a specific example.
        """
        
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
        """
            Regular training loop.

            Parameters
            ----------
                model_path: file path to save pytorch model.
                n_epochs: number of training epochs.
        """
        best_loss = float("inf")

        self.weight_init(model_path)

        with tqdm.tqdm(range(1,n_epochs), unit = "epoch") as tepoch:
            for epoch in tepoch:

                self.training_logs["epoch"].append(epoch)
                self.training_logs["iter"].append(0)
                self.training_logs["unlabelled_count"].append(torch.sum(self.data["unlabelled_mask"]))

                self.__train_model()
                self.__val_model()
                if self.training_logs["val_loss"][-1]<best_loss:
                    torch.save(self.model.state_dict(), model_path/"best_ind_model.pth")
                    best_loss = self.training_logs["val_loss"][-1]
            
                tepoch.set_postfix(training_loss=self.training_logs["training_loss"][-1], 
                                validation_loss=self.training_logs["val_loss"][-1], 
                                training_accuracy=100. * self.training_logs["training_acc"][-1],
                                validation_accuracy=100. * self.training_logs["val_acc"][-1],
                                )

                    
                    
    def self_supervised_training_loop(self, model_path, n_epochs=101, n_iters=5, threshold = 0.8):
        """
            Semi-supervised training loop.

            Parameters
            ----------
                model_path: file path to save pytorch model.
                n_epochs: number of training epochs.
                n_iters: number of training examples updates
                threshold: probability threshold
        """
        best_loss = float("inf")
        
        self.weight_init(model_path)
        with tqdm.tqdm(range(1,n_iters), unit = "Iteration", position=1, leave=False) as t_iter:
            
            for iter in t_iter:
                with tqdm.tqdm(range(1,n_epochs), unit = "epoch", position=0, leave=False) as tepoch:
                    for epoch in tepoch:

                        self.training_logs["epoch"].append(epoch)
                        self.training_logs["iter"].append(iter)
                        self.training_logs["unlabelled_count"].append(torch.sum(self.data["unlabelled_mask"]))

                        self.__train_model()
                        self.__val_model()
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
        """
        Method for re-inintializing a model.

        Parameters
        ----------
            model_path: file path to save pytorch model.
        """
        if not os.path.exists(model_path/"random_wts.pth"):
            torch.save(self.model.state_dict(), model_path/"random_wts.pth")
        
        self.model.load_state_dict(torch.load(model_path/"random_wts.pth", weights_only=True))
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr = 0.0001)
    
    def predictions(self, mask):
        """
        Method for retrieving model predictions based on training mask.
        """

        X_data = self.normalize(self.data["X"]).to(device)

        predictions = self.model(X_data, self.data["edge_index"].to(device))[self.data[mask]]
        ground_truth = (self.data["y"]).type(torch.float)[self.data[mask]]

        return (predictions>0.5).type(torch.long).cpu().numpy() , ground_truth.cpu().numpy() 