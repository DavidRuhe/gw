from train_flow import load_yaml, pop_configs_from_sys_argv
import os


if __name__ == "__main__":
    (run_config_path,) = pop_configs_from_sys_argv("-C", default=["configs/base.yml"])
    run_config = load_yaml(run_config_path)
    os.system(run_config["command"] + " -C " + run_config_path)