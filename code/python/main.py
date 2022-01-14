import argparse
import experiments
import inspect


expsNames = [en for en, _ in inspect.getmembers(experiments, inspect.isclass)]

parser = argparse.ArgumentParser(description='Parse arguments for the CONVERSATIONAL-framework.')
parser.add_argument('experiment', help="experiment to be run")

parser.add_argument('-c', '--collectionId', type=str,
                    help="collection to be used (i.e., CAsT).",
                    choices=['CAsT'], default='CAsT')

parser.add_argument('-dbg', '--debug', type=bool,
                    help="debug: (True|False)",
                    default=True)


parser.add_argument('-pr', '--processors', type=int,
                    help="number of processors",
                    default=1)

params = vars(parser.parse_args())

if params['experiment'] in expsNames:
    experiment = eval(f"experiments.{params['experiment']}(**params)")
    experiment.run_experiment()
else:
    raise NotImplementedError(f"required experiment {params['experiment']} not recognized")