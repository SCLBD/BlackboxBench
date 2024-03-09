import torch
import numpy as np
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def flatness_visualization(args, batch, ori_imgs, adv_imgs, true_labels, target_labels, ensemble_models, loss_func):

    n_plot = 10  # the number of shown examples
    n_model = len(ensemble_models)
    n_rand = 20
    fig = plt.figure(figsize=(3*n_plot, 3*n_model))

    ori_imgs = ori_imgs.detach().clone()
    adv_imgs = adv_imgs.detach().clone()
    true_labels = true_labels.detach().clone()
    target_labels = target_labels.detach().clone()

    x = np.arange(-20, 20, 0.5)
    for i_img in range(n_plot):
        img = adv_imgs[i_img].unsqueeze(0)
        true_label = true_labels[i_img].unsqueeze(0)
        target_label = target_labels[i_img].unsqueeze(0)
        for j_model in range(n_model):
            ax = fig.add_subplot(n_model, n_plot, i_img + j_model * n_plot + 1)
            flatness_matrix = np.zeros((n_rand, len(x)))
            rand_imgs = []
            for k_rand in range(n_rand):
                torch.manual_seed(args.seed)
                torch.cuda.manual_seed(args.seed)
                d = torch.tensor(torch.randn(img.shape) * 0.1).to(DEVICE)
                d = d / d.norm()
                for step in x:
                    rand_imgs.append((img + step * d))
            rand_imgs = torch.concat(rand_imgs)
            with torch.no_grad():
                loss_adv = loss_func(args, img, true_label, target_label, [ensemble_models[j_model]]).cpu().numpy()
                losses = loss_func(args, rand_imgs, true_label.repeat(rand_imgs.shape[0]),
                                   target_label.repeat(rand_imgs.shape[0]), [ensemble_models[j_model]],
                                   reduction='none').cpu().numpy()
            for k_rand in range(n_rand):
                for v, step in enumerate(x):
                    flatness_matrix[k_rand, v] = losses[k_rand * len(x) + v] - loss_adv
            ax.plot(x, np.mean(flatness_matrix, axis=0))
            ax.tick_params(axis='x', labelsize=7)
            ax.set_yticks([])
    plt.tight_layout()
    # plt.show()
    plt.savefig("flatness_Bayesian{}.png".format(batch))

    # for i_img in range(n_plot):
    #     for j_model in range(n_model):
    #         x = np.arange(-20, 20, 0.5)
    #         loss_matrix = np.zeros((n_rand, len(x)))
    #         ax = fig.add_subplot(n_plot, n_model, i_img * n_model + j_model + 1)
    #         for k_rand in range(n_rand):
    #             img = adv_imgs[i_img].unsqueeze(0)
    #             true_label = true_labels[i_img].unsqueeze(0)
    #             target_label = target_labels[i_img].unsqueeze(0)
    #
    #             d = torch.tensor(torch.randn(img.shape) * 0.1).to(DEVICE)
    #             d = d / d.norm()
    #             for v, step in enumerate(x):
    #                 rand_img = img + step * d
    #                 with torch.no_grad():
    #                     flatness = loss_func(args, rand_img, true_label, target_label, [ensemble_models[j_model]]).cpu().numpy()\
    #                                - loss_func(args, img, true_label, target_label, [ensemble_models[j_model]]).cpu().numpy()
    #                     loss_matrix[k_rand, v] = flatness
    #         ax.plot(x, np.mean(loss_matrix, axis=0))
    # # plt.show()
    # plt.savefig("flatness.png", dpi=300)