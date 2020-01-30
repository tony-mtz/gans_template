import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt

# from utils.display_utils import image_gray

'''
save_best and save_last are paths
'''
def train_loop(
                train_loader, 
                epochs,
                discriminator, 
                generator,
                discriminator_opt,
                generator_opt,
                criterion,
                save_path,
                save_last=True,
                show_img=False
                ):    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  

    d_mean_train_losses = []
    g_mean_train_losses = []
    minDLoss = 99999
    minGLoss = 99999
    fixed_z = None

    #generate a fixed z arr to show improvements
    #over epochs
    if show_img:
        np.random.seed(0)
        fixed_z = np.random.normal(0,1, (1,generator.z))

    for epoch in range(epochs):
        print('EPOCH: ',epoch+1)
                
        d_running_loss = 0.0
        g_running_loss = 0.0
        discriminator.train()
        generator.train()             
        count = 0

        for images in train_loader:   
            
            images = Variable(images).to(device)
            labels = Variable(torch.ones(images.shape[0])).to(device)        
            
            #########################################
            # train discriminator
            #########################################

            #p_i = D(x_i) to y_i =1            
            discriminator_out = discriminator(images)         
            discriminator_loss = criterion(discriminator_out, labels)
            
            #train Discriminator with fake data        
            noise = torch.from_numpy(np.random.normal(0,1, (labels.shape[0],100))).float().to(device)
            fake_label = Variable(torch.zeros(images.shape[0])).to(device)
                       
            #p_i = D(G(z_i)) to y_i =0
            d_fake_results = discriminator(generator(noise))
            d_fake_loss = criterion(d_fake_results, fake_label)
            
            discriminator_losses = discriminator_loss + d_fake_loss
            d_running_loss += discriminator_losses.item()
            
            discriminator.zero_grad()  
            discriminator_losses.backward()
            discriminator_opt.step()              
            
            #########################################
            # train generator
            #########################################
            
            noise = torch.from_numpy(np.random.normal(0,1, (labels.shape[0],100))).float().to(device)
            
            d_result = discriminator(generator(noise))
            g_loss = criterion(d_result, labels)
            g_running_loss += g_loss.item()
            
            discriminator.zero_grad()
            generator.zero_grad()
            g_loss.backward()
            generator_opt.step()

            count +=1
        
        d_ave = d_running_loss/count
        g_ave = g_running_loss/count
        print('DISCRIMINATOR Training loss:...', d_ave )
        d_mean_train_losses.append(d_ave)
        
        print('GENERATOR Training loss:...', g_ave)
        print('')
        g_mean_train_losses.append(g_ave)
        
        if d_ave < minDLoss:
            torch.save(discriminator.state_dict(), save_path+'best_discr.pth')
            print('Best DLoss : ', d_ave, '....OLD : ', minDLoss)
            minDLoss = d_ave
        
            
        if g_ave < minGLoss:
            torch.save(generator.state_dict(), save_path+'best_gen.pth')
            print('Best GLoss : ', g_ave, '....OLD : ', minGLoss)
            minGLoss = g_ave
        

        if save_last:
            torch.save(generator.state_dict(), save_path+'last_gen.pth' )
        
        #generate an image and display
        if show_img:
            generator.eval()

            noise = torch.from_numpy(fixed_z).float().to(device)
            result = generator(noise)
        #     result.shape
            plt.figure()
            plt.subplots(figsize=(3,3))
            plt.imshow(result[0].squeeze().data.cpu().numpy(), cmap='gray')
            plt.show()
        
        print('') 

    return [d_mean_train_losses, g_mean_train_losses]