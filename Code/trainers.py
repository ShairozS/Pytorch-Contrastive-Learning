import torch
import numpy as np
import tqdm


class Trainer:

    ##########################
    #                        #
    #     Initialization     #
    #                        #
    ##########################



    def __init__(self,
                 model,
                 dataloader,
                 optimizer = None,
                 lr = 0.0005,
                 loss_function=None,
                 device='cuda'):
        

        
        ## Assign class attributes and send model to device
        self.model = model.to(device)
        self.dataloader = dataloader
        
        
        ## Initialize the optimizer if none has been passed in
        if optimizer is None:
            self.optimizer = torch.optim.Adam(model.parameters(), lr = lr)
        else:
            self.optimizer = optimizer


        ## Initialize the loss function(s)
        self.loss_function = loss_function
        
        self.device = device
        
        self.curr_epoch = 0

    ##########################
    #                        #
    #  Single Iter Training  #
    #                        #
    ##########################


    def train_iter(self, x1, x2, verbose=0):

        ## Zero the gradients
        self.optimizer.zero_grad()

        ## Pass the inputs through the model
        emb1 = self.model(x1)
        emb2 = self.model(x2)


        ## Calculate the loss(es)
        loss = self.loss_function(emb1, emb2)
        

        ## Pass the loss backward
        loss.backward()

        ## Take an optimizer step
        self.optimizer.step()

        ## Return the total loss
        return(loss)

    ##########################
    #                        #
    #  Mutlti Epoch Training #
    #                        #
    ##########################



    def train(self,
              epochs,
              print_every=1,
              writer=None):

        
        ## Loop over epochs in the range of epochs
        epoch_losses = []

        for epoch in range(self.curr_epoch, self.curr_epoch + epochs):
            

            ## If the report_every epoch is reached, reinitialize metric lists
            if epoch % print_every == 0:
                print("----- Epoch: " + str(epoch) + " -----")
            
    
            ## Enumerate self.dataloader
            #tot_iter = self.dataloader.dataset.__len__() // self.dataloader.batch_size
            
            #for idx, data_dict in enumerate(tqdm(self.dataloader, total=tot_iter)):
            batch_losses = []

            for idx, data_dict in tqdm.tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
                
                ## Grab an example
                x1 = data_dict["x1"]; x2 = data_dict["x2"]
                
                
                ## Send it to self.device
                x1 = x1.to(self.device); x2 = x2.to(self.device)
                
                
                ## Try to train_iter
                batch_loss = self.train_iter(x1, x2)


                ## Update the metric lists and counters
                batch_losses.append(batch_loss.item())
            
            
            epoch_losses.append(np.mean(batch_losses))
            self.curr_epoch += 1
            
    
            ## If we've hit report_every epoch, print the report
            if epoch % print_every == 0:
                print("Avg train loss: " + str(np.mean(epoch_losses)))


                ## Logging
                if writer is not None:
                    pass 
                
        return(epoch_losses)
                
                
                
class Tester:

    ##########################
    #                        #
    #     Initialization     #
    #                        #
    ##########################



    def __init__(self,
                 model,
                 dataloader,
                 metric=None,
                 device='cuda'):
        

        
        ## Assign class attributes and send model to device
        self.model = model.to(device)
        self.dataloader = dataloader
        self.metric = metric
        self.device = device
        self.curr_epoch = 0

        
    ##########################
    #                        #
    #  Single Iter Testing   #
    #                        #
    ##########################


    def test_batch(self, input, label):

        ## Pass the inputs through the model
        x1, x2 = input
        res = self.model.predict(x1,x2)
        
        res = res.detach().cpu()
        label = label.detach().cpu()

        ## Calculate the metric
        metric = self.metric(res.flatten(), label.flatten())

        ## Return the metric
        return(metric)

    ##########################
    #                        #
    #  Mutlti Epoch Testing  #
    #                        #
    ##########################



    def test(self):
        
        batch_metrics = []

        ## Enumerate self.dataloader
        for idx, data_dict in enumerate(self.dataloader):

            ## Grab an example
            x1 = data_dict["x1"]; x2 = data_dict["x2"]
            y = data_dict["label"]

            ## Send it to self.device
            x1 = x1.to(self.device); x2 = x2.to(self.device)
            y = y.to(self.device)

            ## Try to train_iter
            batch_metric = self.test_batch((x1,x2), y)

            ## Update the metric lists and counters
            batch_metrics.append(batch_metric.item())
            
        return(np.mean(batch_metrics))