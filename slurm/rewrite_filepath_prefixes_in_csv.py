import pandas 
import pathlib
import argparse

def get_argumentparser():
	parser = argparse.ArgumentParser()
	parser.add_argument('-f', '--file', required=True, type=pathlib.Path, help='CSV file to be rewritten')
	parser.add_argument('-p', '--prefix', required=True, type=pathlib.Path, help='Prefix to be assigned instead of current prefixes')
	parser.add_argument('-o', '--output', required=True, type=pathlib.Path, help='Output file')
	return parser

def main(args):
	df = pandas.read_csv(
		args.file,
		sep=' ',
		names=['filepath', 'label'],
		header=None
	)

	df['filepath'] = df['filepath'].apply(lambda x: args.prefix / pathlib.Path(x).name)
	df.to_csv(args.output, sep=' ', index=False, header=False)
	print(f'Rewritten {args.file} to {args.output} with prefix {args.prefix}.')

if __name__ == '__main__':
	args = get_argumentparser().parse_args()
	main(args)