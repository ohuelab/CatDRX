import os
import torch

def write_continue_training(args):
    with open('main_pretraining_'+args.time+'.sh', "w") as f:
        f.write("""...""")
    

def save_model(output_model_dir, loss, ae, nn=None, save_best=True, epoch=None):
    if not os.path.exists(output_model_dir):
        os.makedirs(output_model_dir)

    if not output_model_dir == '':
        if save_best:
            optimal_loss = loss
            # print('save model with optimal loss: {:.5f}'.format(optimal_loss))
            torch.save(ae.state_dict(), output_model_dir + '/model_ae.pth')
            if nn is not None:
                torch.save(nn.state_dict(), output_model_dir + '/model_nn.pth')
        else:
            current_loss = loss
            # print('save model with current loss: {:.5f}'.format(current_loss))
            torch.save(ae.state_dict(), output_model_dir + '/model_ae_'+str(epoch)+'.pth')
            if nn is not None:
                torch.save(nn.state_dict(), output_model_dir + '/model_nn_'+str(epoch)+'.pth')
    return

def save_model_latest(output_model_dir, optimizer, lossclass, loss, ae, nn=None, epoch=None):
    if not os.path.exists(output_model_dir):
        os.makedirs(output_model_dir)

    if not output_model_dir == '':
        # print('save latest model: {:d}'.format(epoch))
        torch.save({'epoch': epoch,
            'model_state_dict': ae.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'loss_class': lossclass,
            }, output_model_dir + '/model_ae_latest.pth')
        if nn is not None:
            torch.save({'epoch': epoch,
                'model_state_dict': nn.state_dict()
                }, output_model_dir + '/model_nn_latest.pth')
    return

def save_model_latest_temp(output_model_dir, optimizer, lossclass, loss, ae, nn=None, epoch=None):
    if not os.path.exists(output_model_dir):
        os.makedirs(output_model_dir)

    if not output_model_dir == '':
        # print('save latest model: {:d}'.format(epoch))
        torch.save({'epoch': epoch,
            'model_state_dict': ae.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'loss_class': lossclass,
            }, output_model_dir + '/model_ae_latest_temp.pth')
        if nn is not None:
            torch.save({'epoch': epoch,
                'model_state_dict': nn.state_dict()
                }, output_model_dir + '/model_nn_latest_temp.pth')
    return

def save_report(output_model_dir, epoch, loss_train, loss_val, optimal_loss, validity, diversity):
    if not os.path.exists(output_model_dir):
        os.makedirs(output_model_dir)

    if not output_model_dir == '':
        with open(output_model_dir + '/report.txt', "a") as f:
            f.write('epoch: {:d}\tt_loss: {:.5f}\tv_loss: {:.5f}\topt_loss: {:.5f}\tvalid: {:.5f}\tdiver: {:.5f}\n'
                    .format(epoch, loss_train, loss_val, optimal_loss, validity, diversity))
    return

def save_loss(output_model_dir, epoch, all_loss_train, all_loss_val, annealing_step, slope):
    if not os.path.exists(output_model_dir):
        os.makedirs(output_model_dir)

    if not output_model_dir == '':
        with open(output_model_dir + '/loss.txt', "a") as f:
            if 'nn_loss' not in all_loss_train: 
                f.write('epoch: {:d}\trecon_t: {:.5f}\tkl_t: {:.5f}\trecon_v: {:.5f}\tkl_v: {:.5f}\tan_step: {:d}\tslop: {:.5f}\n'
                        .format(epoch, 
                                all_loss_train['recon_loss'], all_loss_train['kl_loss'],
                                all_loss_val['recon_loss'], all_loss_val['kl_loss'], 
                                annealing_step, slope))
            else:
                f.write('epoch: {:d}\trecon_t: {:.5f}\tkl_t: {:.5f}\tnn_t: {:.5f}\trecon_v: {:.5f}\tkl_v: {:.5f}\tnn_v: {:.5f}\tan_step: {:d}\tslop: {:.5f}\n'
                        .format(epoch, 
                                all_loss_train['recon_loss'], all_loss_train['kl_loss'], all_loss_train['nn_loss'], 
                                all_loss_val['recon_loss'], all_loss_val['kl_loss'], all_loss_val['nn_loss'], 
                                annealing_step, slope))
    return