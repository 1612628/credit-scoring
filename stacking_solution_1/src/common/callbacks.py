# callbacks
from deepsense import neptune

from keras.callbacks import Callback 

ctx = neptune.Context()

def neptune_monitor_lgbm(channel_prefix=''):
  def callback(env):
    for name, loss_name, loss_value, _ in env.evaluation_result_list:
      if channel_prefix != '':
        channel_name = f'{channel_prefix}_{name}_{loss_name}'
      else:
        channel_name = f'{name}_{loss_name}'
    ctx.channel_send(channel_name, x=env.iteration, y=loss_value)

  return callback

class NeptuneMonitor(Callback):
  def __init__(self, model_name):
    super().__init__()
    self.model_name_ = model_name
    self.ctx_ = neptune.Context()
    self.epoch_loss_channel_name_ = f'{self.model_name_} Log-loss training'
    self.epoch_val_loss_channel_name_ = f'{self.model_name_} Log-loss validation'

    self.epoch_id_ = 0
    self.batch_id_ = 0

  def on_batch_end(self, batch, logs={}):
    self.batch_id_ += 1
  
  def on_epoch_end(self, epoch, logs={}):
    self.epoch_id_ += 1
    self.ctx_.channel_send(self.epoch_loss_channel_name_, self.epoch_id_, logs['loss'])
    self.ctx_.channel_send(self.epoch_val_loss_channel_name_, self.epoch_id_, logs['loss'])