import argparse
import manage
parser= argparse.ArgumentParser()
parser.add_argument("--path")
parser.add_argument("--phrase")
args=parser.parse_args()
path=args.path
man=manage.Manager.load(path)
print(man.translate(args.phrase))
