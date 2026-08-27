import numpy as np
import argparse

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description='Aggregate numpy arrays.')
        parser.add_argument('--input', nargs='+', help='Input numpy array files to aggregate.')
        parser.add_argument('--output', nargs=1, help='Output file for the aggregated numpy array.')
        args = parser.parse_args()
        input_files = args.input
        output_file = args.output
        arrays = [np.load(f) for f in input_files]
        combined = np.concatenate(arrays, axis=0)
        np.save(output_file[0], combined)