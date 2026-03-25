class DebugConfig:
    def __init__(self):
        self.step_time = True
        self.step_batch = False
        self.attention_layer_breakdown = False
        self.management_breakdown = False
        self.hit_log = False
        self.miss_rate = False
        self.miss_rate_avg = 0.0
        self.log_str: str = ""
        self.print_all_log = False
        self.test = False
        self.swap_copy_ops = True

global_debug_config = DebugConfig()

