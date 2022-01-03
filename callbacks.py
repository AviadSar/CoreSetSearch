from transformers import TrainerCallback, EarlyStoppingCallback


class StopEachEpochCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, logs=None, **kwargs):
        control.should_training_stop = True


class EvaluateAndSaveEpochCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, logs=None, **kwargs):
        control.should_evaluate = True
        control.should_save = True

class EvaluateAndSaveCallback(TrainerCallback):
    def on_step_end(self, args, state, control, logs=None, **kwargs):
        if state.global_step % args.eval_steps == 0:
            control.should_evaluate = True
            control.should_save = True


class LoggingCallback(TrainerCallback):
    def __init__(self, args):
        super.__init__()
        self.args = args

    def on_step_end(self, callback_args, state, control, logs=None, **kwargs):
        if state.global_step % self.args.logging_steps == 0:
            control.should_log = True