import configs
import importlib
import simulators


def main(cfg):
    simulator_cfg = cfg.simulator
    simulator_class = getattr(simulators, simulator_cfg.__dict__.pop("name"))
    simulator = simulator_class(**simulator_cfg.__dict__)
    simulator.run()




if __name__ == "__main__":
    cfg = configs.parse_cfg()
    main(cfg)
