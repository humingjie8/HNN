import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=256, help='testing size')

    # To test changing the training method, also update the weights path and the method below.
    parser.add_argument('--weights_dir', type=str, default=r'snapshots_polyp_gen\PraNet_Res2Net\HNN\20260116_192919', help='directory containing weight files')
    parser.add_argument('--pth_path', type=str, default='')
    parser.add_argument('--approches', type=str,
                        default='EWC')
    # ... other params you may need ...
    return parser