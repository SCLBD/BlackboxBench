import argparse

def parse_handle():
    '''
    input hyper parameters
    '''
    parser = argparse.ArgumentParser()

    

    parser.add_argument('--source_model', type=str, default='resnet50', choices=['resnet50', 'inception-v3', 'densenet121', 'vgg16bn'])

    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--max_iterations', type=int, default=20)

    parser.add_argument('--loss_function', type=str, default='CE', choices=['CE','MaxLogit'])

    parser.add_argument('--targeted', action='store_true')

    parser.add_argument('--m1', type=int, default=1, help='number of randomly sampled images')
    parser.add_argument('--m2', type=int, default=1, help='num of copies')
    parser.add_argument('--strength', type=float, default=0)

    parser.add_argument('--gaussian_noise', action='store_true')
    parser.add_argument('--adv_perturbation', action='store_true')

    parser.add_argument('--adv_loss_function', type=str, default='CE', choices=['CE', 'MaxLogit'])
    parser.add_argument('--adv_targeted', action='store_true')
    parser.add_argument('--adv_epsilon', type=eval, default=16/255)
    parser.add_argument('--adv_steps', type=int, default=8)
    parser.add_argument('--adv_label', type=str, default='target', choices=['pred', 'target'])

    parser.add_argument('--transpoint', type=int, default=0)

    parser.add_argument('--MI', action='store_true')
    parser.add_argument('--DI', action='store_true')
    parser.add_argument('--TI', action='store_true')
    parser.add_argument('--NI', action='store_true')
    parser.add_argument('--SI', action='store_true')

    return parser