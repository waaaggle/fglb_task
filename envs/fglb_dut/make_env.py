from fglb_task.envs.fglb_dut.fglb_wrapper import FglbWrapper

def make_fglb_env(all_args, rank):
    def _thunk():
        return FglbWrapper(all_args)
    return _thunk
