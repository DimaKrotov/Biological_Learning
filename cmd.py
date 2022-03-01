import argparse

args = None

def parse_arguments():

    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument('-B', '--batch_size', default=1000, type=int, help="Batch size")
    parser.add_argument('-E', '--epochs', default=100, type=int, help="Number of Epochs")
    parser.add_argument('--lr', '--learning_rate', default=0.001, type=float, help="Learning rate")
    parser.add_argument('--sess', '--session_name', default="MNIST", type=str, help="Session name")

    args = parser.parse_args()
    return args

# Parse arguments
def run_args():
    global args
    if args is None:
        args = parse_arguments()

run_args()
