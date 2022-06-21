import configs
import importlib
from configs.parse import parse_args
import simulators


def main(args):
    simulator_class = getattr(simulators, args['simulator'].pop('object'))
    simulator = simulator_class(**args['simulator'])
    simulator.run()




if __name__ == "__main__":
    args = parse_args()
    main(args)
