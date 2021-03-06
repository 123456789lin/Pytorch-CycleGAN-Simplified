
import importlib
from models.base_model import BaseModel
from models.cycle_gan_model import CycleGANModel
from models.pix2pix_model import Pix2PixModel


def get_option_setter(model_name):
    """Return the static method <modify_commandline_options> of the model class."""
    model_class = CycleGANModel if model_name == 'cycle_gan' else Pix2PixModel
    return model_class.modify_commandline_options

def create_model(opt):
    model = CycleGANModel if opt.model == 'cycle_gan' else Pix2PixModel
    instance = model(opt)
    print("model [%s] was created" % type(instance).__name__)
    return instance
