from sklearn.metrics import accuracy_score

import tqdm
import torch
import torch.nn.functional as F

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

        # model
        self.model = model.to(device)

        self.data = data

        self.optimizer = torch.optim.Adam(self.model.parameters(),lr = 0.001)

        self.t_means = (self.data["X"]*self.get_mask_for_X("train_mask")).mean(dim=0, keepdim=True)
        self.t_stds = (self.data["X"]*self.get_mask_for_X("train_mask")).std(dim=0, keepdim=True)
        

    def get_mask_for_X(self, mask_name):

        return self.data[mask_name].type(torch.long).reshape(-1,1).repeat(1,self.data["X"].shape[1])
    
    def normalize(self, data):

        normalized_data = (data - self.t_means) / (self.t_stds+1e-10)
        return normalized_data

    def train_model(self):
        

        torch.manual_seed(47)
            
        self.model.train()
        # Sets the gradients of all optimized tensors to zero
        self.optimizer.zero_grad()
        
        X_data = self.normalize(self.data["X"]*self.get_mask_for_X("train_mask")).to(device)

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
            predictions = self.model(self.normalize(self.data["X"]*self.get_mask_for_X("val_mask")).to(device), self.data["edge_index"].to(device))[self.data["val_mask"]]
            ground_truth = (self.data["y"]).type(torch.float)[self.data["val_mask"]]
            loss = torch.nn.BCEWithLogitsLoss()(torch.reshape(predictions,(-1,)), ground_truth).to(device)
        
        # metrics
        self.training_logs["val_loss"].append(loss.item())
        val_accuracy = accuracy_score(ground_truth.cpu().numpy(), (predictions>0.5).type(torch.long).cpu().numpy() )
        self.training_logs ["val_acc"].append(val_accuracy)


    def update_training_and_unlabelled_mask(self):

        # need to change to Crosee Entropy loss for easier stuff
        pass

    def training_loop(self, model_path, n_epochs=101, n_iters=5):
        best_loss = float("inf")

        with tqdm.tqdm(range(1,n_iters), unit = "epoch") as t_iter:
            
            for iter in t_iter:

                with tqdm.tqdm(range(1,n_epochs), unit = "epoch") as tepoch:
                    for epoch in tepoch:

                        self.training_logs["epoch"].append(epoch)
                        
                        self.train_model()
                        self.val_model()
                        if self.training_logs["val_loss"][-1]<best_loss:
                            torch.save(self.model.state_dict(), model_path)
                            best_loss = self.training_logs["val_loss"][-1]
                        
                        tepoch.set_postfix(training_loss=self.training_logs["training_loss"][-1], 
                                        validation_loss=self.training_logs["val_loss"][-1], 
                                        training_accuracy=100. * self.training_logs["training_acc"][-1],
                                        validation_accuracy=100. * self.training_logs["val_acc"][-1],
                                        )
                t_iter.set_postfix(training_loss=self.training_logs["training_loss"][-1], 
                                        validation_loss=self.training_logs["val_loss"][-1], 
                                        training_accuracy=100. * self.training_logs["training_acc"][-1],
                                        validation_accuracy=100. * self.training_logs["val_acc"][-1],
                                        )
    
        

            
            
            
           