import os
import sys
import argparse
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

def csv2tboard(csv_path, log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    writer = SummaryWriter(log_dir)

    df = pd.read_csv(csv_path)
    df['cum_steps'] = df['l'].cumsum()
    for idx, row in df.iterrows():
        writer.add_scalar('episode_reward', row['r'], row['cum_steps'])

    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default='./metrics/monitor.csv', help='Path to the CSV file.')
    parser.add_argument('--log_dir', type=str, default='./metrics', help='Path to the log directory.')
    args = parser.parse_args()

    csv2tboard(args.csv_path, args.log_dir)
