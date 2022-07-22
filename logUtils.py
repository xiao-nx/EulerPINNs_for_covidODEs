


def print_training(epoch, run_time, tmp_lr, penalty_wb2beta, penalty_wb2gamma, penalty_wb2mu, loss_s, loss_i,
                        loss_r, loss_d, loss_all, log_out=None):
    print('train epoch: %d, time: %.4f' % (epoch, run_time))
    print('learning rate: %.10f' % tmp_lr)
    print('penalty weights and biases for Beta: %.8f' % penalty_wb2beta)
    print('penalty weights and biases for Gamma: %.8f' % penalty_wb2gamma)
    print('penalty weights and biases for Mu: %.8f' % penalty_wb2mu)
    print('loss for S: %.8f' % loss_s)
    print('loss for I: %.8f' % loss_i)
    print('loss for R: %.8f' % loss_r)
    print('loss for D: %.8f' % loss_d)
    print('total loss: %.8f\n' % loss_all)

    log_out.write('train epoch: %d,time: %.4f \n' % (epoch, run_time))
    log_out.write('learning rate: %.8f \n' % tmp_lr)
    log_out.write('penalty weights and biases for Beta: %.8f \n' % penalty_wb2beta)
    log_out.write('penalty weights and biases for Gamma: %.8f \n' % penalty_wb2gamma)
    log_out.write('penalty weights and biases for Mu: %.8f \n' % penalty_wb2mu)
    log_out.write('loss for S: %.8f \n' % loss_s)
    log_out.write('loss for I: %.8f \n' % loss_i)
    log_out.write('loss for R: %.8f \n' % loss_r)
    log_out.write('loss for D: %.8f \n' % loss_d)
    log_out.write('total loss: %.8f \n\n' % loss_all)
    log_out.flush()   # 清空缓存区

