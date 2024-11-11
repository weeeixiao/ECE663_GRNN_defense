import argparse
import time, datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from grnn_utils import *
from Generator.model import Generator
from TFLogger.logger import TFLogger
from Backbone import *
torch.set_default_tensor_type('torch.cuda.FloatTensor')


def get_args():
    parser = argparse.ArgumentParser(description="GRNN Configuration")

    # GPU and device configurations
    parser.add_argument("--cuda_device_order", type=str, default="PCI_BUS_ID", help="Order of CUDA device.")
    parser.add_argument("--cuda_visible_devices", type=str, default="0", help="CUDA visible devices.")
    parser.add_argument("--device0", type=int, default=0, help="Device for GRNN training.")
    parser.add_argument("--device1", type=int, default=0, help="Device for local training.")

    # Training parameters
    parser.add_argument("--batchsize", type=int, default=1, help="Batch size for training.")
    parser.add_argument("--save_img", type=bool, default=True, help="Whether to save generated images.")
    parser.add_argument("--iteration", type=int, default=1000, help="Number of optimization steps on GRNN.")
    parser.add_argument("--num_exp", type=int, default=10, help="Experiment number.")
    parser.add_argument("--g_in", type=int, default=128, help="Dimension of GRNN input.")
    parser.add_argument("--plot_num", type=int, default=30, help="Number of plots to generate.")

    # Model and dataset configurations
    parser.add_argument("--net_name", type=str, default="lenet", choices=['lenet', 'res18'], help="Global model name.")
    parser.add_argument("--dataset", type=str, default="lfw", choices=['mnist', 'cifar100', 'lfw', 'VGGFace', 'ilsvrc'], help="Dataset name.")
    parser.add_argument("--shape_img", type=tuple, default=(32, 32), help="Shape of the images.")

    # Paths
    parser.add_argument("--root_path", type=str, default='./', help="Root path for the project.")
    parser.add_argument("--data_path", type=str, default='./data/', help="Path for the data.")

    return parser.parse_args()


def main(args):
    os.environ["CUDA_DEVICE_ORDER"] = args.cuda_device_order
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    # Define save paths
    save_path = os.path.join(
        args.root_path,
        f"Results/GRNN-{args.net_name}-{args.dataset}-S{args.shape_img[0]}-B{str(args.batchsize).zfill(3)}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}/"
    )
    save_img_path = os.path.join(save_path, 'saved_img/')
    print('>' * 10, save_path)

    # Load dataset, transform and dataloader
    dst, num_classes = gen_dataset(args.dataset, args.data_path, args.shape_img)
    tp = transforms.Compose([transforms.ToPILImage()])
    train_loader = iter(torch.utils.data.DataLoader(dst, batch_size=args.batchsize, shuffle=True, generator=torch.Generator(device='cuda'),))

    # Define loss function
    criterion = nn.CrossEntropyLoss().cuda(args.device1)

    print(f'{str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))}: {save_path}')

    # -----------------------------------------------------------------------------------------------
    #                                     begin of experiment
    # -----------------------------------------------------------------------------------------------
    for idx_net in range(args.num_exp):
        # train_tfLogger = TFLogger(f'{args.save_path}/tfrecoard-exp-{str(idx_net).zfill(2)}') 
        print(f'{time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime())}: Running experiment {idx_net + 1}/{args.num_exp}')

        # Claim global model and attack model
        net = LeNet(num_classes=num_classes) if args.net_name == 'lenet' else ResNet18(num_classes=num_classes)
        net = net.cuda(args.device1)
        Gnet = Generator(num_classes, channel=3, shape_img=args.shape_img[0], batchsize=args.batchsize, g_in=args.g_in).cuda(args.device0)

        # Initialize
        net.apply(weights_init)
        Gnet.weight_init(mean=0.0, std=0.02)
        G_optimizer = torch.optim.RMSprop(Gnet.parameters(), lr=0.0001, momentum=0.99)
        tv_loss = TVLoss()

        # Calculate TRUE gradient
        gt_data, gt_label = next(train_loader)
        gt_data, gt_label = gt_data.cuda(args.device1), gt_label.cuda(args.device1)
        pred = net(gt_data)
        y = criterion(pred, gt_label)
        dy_dx = torch.autograd.grad(y, net.parameters())
        flatten_true_g = flatten_gradients(dy_dx)

        # Initialize input for attack model
        G_ran_in = torch.randn(args.batchsize, args.g_in).cuda(args.device0)
        iter_bar = tqdm(range(args.iteration), total=args.iteration, desc=f'{time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime())}', ncols=180, dynamic_ncols=True, leave=True)
        
        # -----------------------------------------------------------------------------------------------
        #                                Optimize for the attack model
        # -----------------------------------------------------------------------------------------------
        history, history_l = [], []
        for iters in iter_bar:
            Gout, Glabel = Gnet(G_ran_in)
            Gout, Glabel = Gout.cuda(args.device1), Glabel.cuda(args.device1)
            Gpred = net(Gout)
            
            # Calculate fake gradient
            Gloss = -torch.mean(torch.sum(Glabel * torch.log(torch.softmax(Gpred, 1)), dim=-1))
            G_dy_dx = torch.autograd.grad(Gloss, net.parameters(), create_graph=True)
            flatten_fake_g = flatten_gradients(G_dy_dx).cuda(args.device1)
            
            grad_diff_l2 = loss_f('l2', flatten_fake_g, flatten_true_g, args.device1)
            grad_diff_wd = loss_f('wd', flatten_fake_g, flatten_true_g, args.device1)
            tvloss = (1e-3 if args.net_name == 'lenet' else 1e-6) * tv_loss(Gout)
            grad_diff = grad_diff_l2 + grad_diff_wd + tvloss

            # Update the generator model
            G_optimizer.zero_grad()
            grad_diff.backward()
            G_optimizer.step()

            iter_bar.set_postfix(loss_l2=round(grad_diff_l2.item(), 8), loss_wd=round(grad_diff_wd.item(), 8), loss_tv=round(tvloss.item(), 8),
                                img_mses=round(torch.mean(abs(Gout - gt_data)).item(), 8),
                                img_wd=round(wasserstein_distance(Gout.view(1, -1), gt_data.view(1, -1)).item(), 8))

            # Save the history of generating figures
            if iters % (args.iteration // args.plot_num) == 0:
                history.append([tp(Gout[imidx].detach().cpu()) for imidx in range(args.batchsize)])
                history_l.append([Glabel.argmax(dim=1)[imidx].item() for imidx in range(args.batchsize)])

            # Clear up the space
            torch.cuda.empty_cache()
            del Gloss, G_dy_dx, flatten_fake_g, grad_diff_l2, grad_diff_wd, grad_diff, tvloss

        # -----------------------------------------------------------------------------------------------
        #                                     Visualize the result
        # -----------------------------------------------------------------------------------------------
        for imidx in range(args.batchsize):
            plt.figure(figsize=(12, 8))
            plt.subplot(args.plot_num // 10, 10, 1)
            plt.imshow(tp(gt_data[imidx].cpu()))
            for i in range(min(len(history), args.plot_num - 1)):
                plt.subplot(args.plot_num // 10, 10, i + 2)
                plt.imshow(history[i][imidx])
                plt.title(f'l={history_l[i][imidx]}')
                plt.axis('off')

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # Save the true and fake figures
            if args.save_img:
                true_path = os.path.join(save_img_path, f'true_data/exp{str(idx_net).zfill(3)}/')
                fake_path = os.path.join(save_img_path, f'fake_data/exp{str(idx_net).zfill(3)}/')
                os.makedirs(true_path, exist_ok=True)
                os.makedirs(fake_path, exist_ok=True)
                tp(gt_data[imidx].cpu()).save(os.path.join(true_path, f'{imidx}_{gt_label[imidx].item()}.png'))
                history[-1][imidx].save(os.path.join(fake_path, f'{imidx}_{Glabel.argmax(dim=1)[imidx].item()}.png'))

            plt.savefig(f'{save_path}/exp:{idx_net:03d}-imidx:{imidx:02d}-tlabel:{gt_label[imidx].item()}-Glabel:{Glabel.argmax(dim=1)[imidx].item()}.png')
            plt.close()

        # Clear up the space
        del Glabel, Gout, flatten_true_g, G_ran_in, net, Gnet
        torch.cuda.empty_cache()
        history.clear()
        history_l.clear()
        iter_bar.close()
        # train_tfLogger.close()
        print('----------------------')



if __name__ == '__main__':
    args = get_args()
    main(args)