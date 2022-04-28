import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from os import path
from .data import ODE_data_generator, get_generator
from .model import neural_net


# residual neural network
@tf.function
def r_PINN(model,param,x,t):
    u    = model(tf.concat([x,t], 1))
    u_x  = tf.gradients(u,x)[0]
    u_xx = tf.gradients(u_x, x)[0]
    u_t  = tf.gradients(u,t)[0]
    return u_t + u*u_x - param*u_xx

# PINN loss function
def loss(model,param,xcl,tcl,xs,ts,us,xlb,tlb,ulb,xub,tub,uub):
    u_pred  = model(tf.concat([xs,ts],1))
    ulb_pred = model(tf.concat([xlb,tlb],1))
    uub_pred = model(tf.concat([xub,tub],1))
    r_pred   = r_PINN(model,param,xcl,tcl)
    # loss components
    mse_s  = tf.reduce_mean(tf.pow(u_pred-us,2)) # data points
    mse_lb = tf.reduce_mean(tf.pow(ulb_pred-ulb,2)) # BC
    mse_ub = tf.reduce_mean(tf.pow(uub_pred-uub,2)) # BC
    mse_r  = tf.reduce_mean(tf.pow(r_pred,2)) # PDE residual loss
    return  mse_s,mse_r,mse_lb,mse_ub

# neural network weight gradients: trainable_variables and param
# @tf.function
# def grad(model,param,*args):
#     with tf.GradientTape(persistent=True) as tape:
#         mse_s,mse_r,mse_lb,mse_ub = loss(model,param,*args)
#         loss_value = mse_s + mse_r + mse_lb + mse_ub
#         grads = tape.gradient(loss_value,model.trainable_variables)
#         grad_param = tape.gradient(loss_value,param) # call twice tape.gradient
#     return loss_value,grads,grad_param


def train(args):
    # initialize new instance of NN
    u_PINN = neural_net()
    if args.log_dir is not None:
        train_summary_writer = tf.summary.create_file_writer(path.join(args.log_dir, 'train'))

    # initialize parameter estimate
    param = tf.Variable(0.1,trainable=True,dtype=tf.float32) # v: viscosity in Burgers PDE
    # Adam optimizer for neural network weights and parameter
    tf_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001,beta_1=0.99)
    tf_optimizer_param = tf.keras.optimizers.Adam(learning_rate=0.001,beta_1=0.99)

    # norm of high-fidelity approximation
    norm_u = np.linalg.norm(u_flat,2)
    norm_u0 = np.linalg.norm(u0,2)

    for iter in range(args.n_epochs):  
        # compute gradients using AD
        # neural network weight gradients: trainable_variables and param

        with tf.GradientTape(persistent=True) as tape:
            mse_s,mse_r,mse_lb,mse_ub = loss(u_PINN,param,xcl,tcl,xs,ts,us,xlb,tlb,ulb,xub,tub,uub)
            loss_value = mse_s + mse_r + mse_lb + mse_ub
            if args.verbose:
                if ((iter == 0) or ((iter+1) % 100 == 0)):
                    print(
                    f'epoch: {iter}, '
                    f'mse_s, mse_lb, mse_ub, mse_r: {mse_s.numpy(), mse_lb.numpy(), mse_ub.numpy(), mse_r.numpy()}, '
                    f'loss_value: {loss_value.numpy()} '
                    )
            if args.log_dir is not None:
                # MSE_s,MSE_r,MSE_lb,MSE_ub,MSE_loss = [],[],[],[]
                with train_summary_writer.as_default():
                    tf.summary.scalar('mse_s', mse_s.numpy(), step=iter) # numpy scalar
                    tf.summary.scalar('mse_lb', mse_lb.numpy(), step=iter)
                    tf.summary.scalar('mse_ub', mse_ub.numpy(), step=iter)
                    tf.summary.scalar('mse_r', mse_r.numpy(), step=iter)
                    tf.summary.scalar('loss_value', loss_value.numpy(), step=iter)         
                # MSE_s.append(mse_s.numpy())
                # MSE_r.append(mse_r.numpy())
                # MSE_lb.append(mse_lb.numpy())
                # MSE_ub.append(mse_ub.numpy())
                # MSE_loss.append(loss_value.numpy())
            grads = tape.gradient(loss_value,u_PINN.trainable_variables)
            grad_param = tape.gradient(loss_value,param) # call twice tape.gradient for nn weights and PDE param

        # loss_value,grads,grad_param = grad(u_PINN,param,xcl,tcl,xs,ts,us,xlb,tlb,ulb,xub,tub,uub)   
        
        # update neural network weights
        tf_optimizer.apply_gradients(zip(grads,u_PINN.trainable_variables))  
        # update parameter estimate
        tf_optimizer_param.apply_gradients(zip([grad_param],[param]))

        # display intermediate results  
        if args.show_steps:
            if ((iter == 0) or ((iter+1) % 500 == 0)):
                print('iter =  '+str(iter+1))
                print('loss = %.4e' % loss_value)
                print('diffusion coefficient estimate = {:.4f}/pi'.format(np.pi*param.numpy()))
                print('L2 error for parameter: %.4e' % (np.abs(param-args.param/np.pi)/(args.param/np.pi)))
                u0_pred = u_PINN(tf.concat([x0,t0],1))
                err0 = np.linalg.norm(u0-u0_pred,2)/norm_u0
                print('L2 error for initial condition: %.4e' % (err0))
                u_PINN_flat = u_PINN(X_flat)
                
                err = np.linalg.norm(u_flat-u_PINN_flat[:,-1],2)/norm_u
                print('L2 error: %.4e' % (err))
                #plot_slices(u_flat,u_PINN_flat,[0.15,0.5,0.85])
                fig = plt.figure(figsize=(12,4),dpi=75)
                plt.style.use('seaborn')
                for gi,snap in enumerate([0.15,0.5,0.85]):
                    tind = int(snap*len(ut))
                    ax = fig.add_subplot(1,3,gi+1)
                    ax.set_aspect(0.5)
                    ax.plot(ux,u_flat[tind*uxn:(tind+1)*uxn],'b-',linewidth=2,label='Exact')       
                    ax.plot(ux,u_PINN_flat[tind*uxn:(tind+1)*uxn,0],'r--',linewidth=2,label='Prediction')
                    ax.set_title('$t = %.2f$' % (ut[tind]),fontsize=10)
                    ax.set_xlabel('$x$')
                    ax.set_ylim([-1.3,1.3])
                    # plt.show()
                    os.makedirs('train_imgs', exist_ok = True)
                    plt.savefig(path.join('train_imgs', f'epoch_{iter}.png'), bbox_inches='tight')
                    plt.clf()
                    
if __name__ == '__main__':
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    parser.add_argument('-n', '--n_epochs', type = int, default = 20000)
    parser.add_argument('-v', '--verbose', action = 'store_true')
    parser.add_argument('-S', '--show_steps', action = 'store_true')
    parser.add_argument('-nu', '--param', type = float, default = 0.01) # nu/np.pi
    parser.add_argument('-Ns', '--n_samples', type = int, default = 10000)
    args = parser.parse_args()
    
    tf.random.set_seed(1234)
    # training and testing points, solutions:
    gen = ODE_data_generator(args.param/np.pi)
    [xcl,tcl,xs,ts,us,xlb,tlb,ulb,xub,tub,uub,x0,t0,u0], [X_flat, u_flat] = get_generator(gen, Ns=args.n_samples, Ncl=10000, Nlb=500, Nub=500, N0=500) # global var
    # for vis:
    uxn = gen.uxn
    ux = np.linspace(gen.xlo,gen.xhi,gen.uxn)
    ut = np.linspace(gen.tlo,gen.thi,gen.utn)

    log_file = open('message.log','w')  
    sys.stdout = log_file
    train(args)
    log_file.close()
    sys.exit()
    # python -m Burgers.train --log_dir logdir/train1 -nu 0.01 -Ns 10000 -S -v