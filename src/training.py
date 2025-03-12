import time
from datetime import datetime

import numpy as np
import scipy.io as scio

import config as cf
from discriminator.discriminator import Discriminator
from generator.ansatz import construct_qcircuit_ZZ_XZ
from generator.generator import Generator
from optimization.cost_and_fidelity import compute_cost, compute_fidelity
from target.target_hamiltonian import construct_target
from target.target_state import get_maximally_entangled_state
from tools.plot_hub import plt_fidelity_vs_iter
from tools.qcircuit import I, Identity, X, Y, Z
from tools.utils import (
    save_fidelity_loss,
    save_model,
    save_theta,
    train_log,
)

np.random.seed()


##### main


def main():
    # preparation for Choi state
    input_state = get_maximally_entangled_state(cf.system_size)
    # input_state = get_maximally_entangled_state_in_subspace(cf.system_size) # 2*(size+1), B0, B1, A0, A1

    # define target gates
    # target_unitary = scio.loadmat('./exp_ideal_{}_qubit.mat'.format(cf.system_size))['exp_ideal']
    target_unitary = construct_target(cf.system_size)
    # target_unitary = construct_clusterH(cf.system_size)
    # target_unitary = construct_RotatedSurfaceCode(cf.system_size)

    # define target state
    real_state = np.matmul(np.kron(target_unitary, Identity(cf.system_size)), input_state)
    # real_state = np.matmul(np.kron(np.kron(np.kron(target_unitary, Identity(1)), Identity(cf.system_size)), Identity(1)), input_state)

    # define generator
    gen = Generator(cf.system_size)
    # gen.set_qcircuit(construct_qcircuit_XYZandfieldZ(gen.qc, cf.system_size, cf.layer))
    gen.set_qcircuit(construct_qcircuit_ZZ_XZ(gen.qc, cf.system_size, cf.layer))

    # define discriminator
    herm = [I, X, Y, Z]
    dis = Discriminator(herm, cf.system_size * 2)
    # dis = Discriminator(herm, (cf.system_size+1)*2)

    # compute fidelity at initial
    f = compute_fidelity(gen, input_state, real_state, input_state)

    fidelities = np.zeros(cf.epochs)
    losses = np.zeros(cf.epochs)
    fidelities_history = []
    losses_history = []
    starttime = datetime.now()
    num_epochs = 0

    while f < 0.99:
        # while (f < 0.95):
        fidelities[:] = 0.0
        losses[:] = 0.0
        num_epochs += 1
        for iter in range(cf.epochs):
            print("==================================================")
            print("Epoch {}, Step_size {}".format(iter + 1 + (num_epochs - 1) * cf.epochs, cf.eta))

            # Generator gradient descent
            gen.update_gen(dis, real_state, input_state)
            # print("Loss after generator step: {}".format(compute_cost(gen, dis, real_state, input_state)))

            # Discriminator gradient ascent
            dis.update_dis(gen, real_state, input_state)
            # print("Loss after discriminator step: {}".format(compute_cost(gen, dis, real_state, input_state)))

            fidelities[iter] = compute_fidelity(gen, input_state, real_state)
            losses[iter] = compute_cost(gen, dis, real_state, input_state)

            print("Fidelity between real and fake state: {}".format(fidelities[iter]))
            print("==================================================")

            if iter % 10 == 0:
                # save log
                endtime = datetime.now()
                training_duration = (endtime - starttime).seconds / float(3600)
                param = "epoches:{:4d} | fidelity:{:8f} | time:{:10s} | duration:{:8f}\n".format(
                    iter,
                    round(fidelities[iter], 6),
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    round(training_duration, 2),
                )
                train_log(param, cf.log_path)

        f = fidelities[-1]
        fidelities_history = np.append(fidelities_history, fidelities)
        losses_history = np.append(losses_history, losses)
        plt_fidelity_vs_iter(fidelities_history, losses_history, cf, num_epochs)

        if num_epochs >= 10:
            print("The number of epochs exceeds 10.")
            break

    # save data of fidelity and loss
    save_fidelity_loss(fidelities_history, losses_history, cf.fid_loss_path)

    # save data of the generator and the discriminator
    save_model(gen, cf.model_gen_path)
    save_model(dis, cf.model_dis_path)

    # output the parameters of the generator
    save_theta(gen, cf.theta_path)

    endtime = datetime.now()
    print("{} seconds".format((endtime - starttime).seconds))
    print("end")


main()
