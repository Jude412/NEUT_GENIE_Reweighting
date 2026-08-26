import numpy as np
import argparse

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description='Aggregate numpy arrays.')
        parser.add_argument('--input', nargs='+', help='Input numpy array files to aggregate.')
        parser.add_argument('--output', nargs=1, help='Output file for the aggregated numpy array.')
        args = parser.parse_args()
        input = args.input
        output = args.output
        arrays = [np.load(f) for f in input]
        combined = np.concatenate(arrays, axis=0)
        np.save(output[0], combined)