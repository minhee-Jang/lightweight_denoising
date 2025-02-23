"""This package contains modules related to objective functions, optimizations, and network architectures.

To add a custom model class called 'dummy', you need to add a file called 'dummy_model.py' and define a subclass DummyModel inherited from BaseModel.
You need to implement the following five functions:
    -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
    -- <set_input>:                     unpack data from dataset and apply preprocessing.
    -- <forward>:                       produce intermediate results.
    -- <optimize_parameters>:           calculate loss, gradients, and update network weights.
    -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.

In the function <__init__>, you need to define four lists:
    -- self.loss_names (str list):          specify the training losses that you want to plot and save.
    -- self.model_names (str list):         define networks used in our training.
    -- self.visual_names (str list):        specify the images that you want to display and save.
    -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an usage.

Now you can use the model class by specifying flag '--model dummy'.
See our template model class 'template_model.py' for more details.
"""

import importlib
from models.base_model import BaseModel
# import torch.nn as nn

def find_model_using_name(model_name):
    """Import the module "models/[model_name].py".

    In the file, the class called DatasetNameModel() will
    be instantiated. It has to be a subclass of BaseModel,
    and it is case-insensitive.
    """
    model_filename = "models." + model_name
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model


def get_option_setter(model_name):
    """Return the static method <modify_commandline_options> of the model class."""
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options


def create_model(opt):
    """Create a model given the option.

    This function warps the class CustomDatasetDataLoader.
    This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from models import create_model
        >>> model = create_model(opt)
    """
    model = find_model_using_name(opt.model)
    instance = model(opt)
    print("model [{}] was created".format(type(instance).__name__))
    return instance

# My function
def get_savedir_setter(model_name):
    """
    Set save directory for checkpoints
    """
    model_class = find_model_using_name(model_name)
    return model_class.set_savedir





# def set_model(opt):
#     if opt.model == 'redcnn':
#         module_name = 'models.redcnn'
#     elif opt.model == 'dncnn':
#         module_name = 'models.dncnn'
#     elif opt.model == 'edsr':
#         module_name = 'models.edsr'
#     elif opt.model == 'erdnet':
#         module_name = 'models.erdnet'
#     elif opt.model == 'eredcnn':
#         module_name = 'models.eredcnn'
#     elif opt.model == 'ridnet':
#         module_name = 'models.ridnet'
#     elif opt.model == 'cbdnet':
#         module_name = 'models.cbdnet'
#     elif opt.model == 'ffdnet':
#         module_name = 'models.ffdnet'
#     elif opt.model == 'waveletdl':
#         if opt.parallel:
#             module_name = 'models.wavelet_par_dl'
#         else:
#             module_name = 'models.wavelet_dl'
#     elif opt.model == 'wavresnet':
#         module_name = 'models.wavresnet'
#     elif opt.model == 'rcan':
#         module_name = 'models.rcan'
#     elif opt.model == 'brdnet':
#         module_name = 'models.brdnet'
#     elif opt.model == 'resbrdnet':
#         module_name = 'models.resbrdnet'
#     elif opt.model == 'wavelet_attn':
#         module_name = 'models.wavelet_attn'
#     elif opt.model == 'mwcnn':
#         module_name = 'models.mwcnn'
#     else:
#         raise ValueError("Need to specify model (redcnn, dncnn)")
    
#     module = import_module(module_name)
#     model = module.make_model(opt)

#     return model

# def forward_ensemble(*args, net=None, device='cuda'):
#     def _transform(v, op):
#         if op == 'v':
#             tf = v.flip(2).clone()
#         elif op == 'h':
#             tf = v.flip(3).clone()
#         elif op == 't':
#             # tf = v.rot90(1, (3, 2)).clone()
#             tf = v.transpose(3, 2).clone()

#         return tf

#     list_x = []
#     for a in args:
#         x = [a]
#         for tf in 'v', 'h', 't': x.extend([_transform(_x, tf) for _x in x])

#         list_x.append(x)
#     # print("len(list_x):", len(list_x))
#     # print("len(list_x[0]):", len(list_x[0]))

#     list_y = []
#     for x in zip(*list_x):
#         y = net(*x)
#         if not isinstance(y, list): y = [y]
#         if not list_y:
#             list_y = [[_y] for _y in y]
#         else:
#             for _list_y, _y in zip(list_y, y): _list_y.append(_y)

#     for _list_y in list_y:
#         for i in range(len(_list_y)):
#             if i > 3:
#                 _list_y[i] = _transform(_list_y[i], 't')
#             if i % 4 > 1:
#                 _list_y[i] = _transform(_list_y[i], 'h')
#             if (i % 4) % 2 == 1:
#                 _list_y[i] = _transform(_list_y[i], 'v')

#     # print("len(list_y):", len(list_y))
#     # print("len(list_y[0]):", len(list_y[0]))
    
#     y = [torch.cat(_y, dim=0).mean(dim=0, keepdim=True) for _y in list_y]

#     # print("y[0].shape:", y[0].shape)

#     if len(y) == 1: y = y[0]

#     return y

# class Model(nn.Module):
#     def __init__(self, opt):
#         super(Model, self).__init__()

#         self.model = set_model(opt)
#         self.ensemble = opt.ensemble
#         self.device = opt.device

#     def forward_ensemble(self, *args, forward_function=None):
#         def _transform(v, op):
#             if op == 'v':
#                 tf = v.flip(2).clone()
#             elif op == 'h':
#                 tf = v.flip(3).clone()
#             elif op == 't':
#                 # tf = v.rot90(1, (3, 2)).clone()
#                 tf = v.transpose(3, 2).clone()

#             return tf

#         list_x = []
#         for a in args:
#             x = [a]
#             for tf in 'v', 'h', 't': x.extend([_transform(_x, tf) for _x in x])

#             list_x.append(x)

#         list_y = []
#         for x in zip(*list_x):
#             y = forward_function(*x)
#             if not isinstance(y, list): y = [y]
#             if not list_y:
#                 list_y = [[_y] for _y in y]
#             else:
#                 for _list_y, _y in zip(list_y, y): _list_y.append(_y)

#         for _list_y in list_y:
#             for i in range(len(_list_y)):
#                 if i > 3:
#                     _list_y[i] = _transform(_list_y[i], 't')
#                 if i % 4 > 1:
#                     _list_y[i] = _transform(_list_y[i], 'h')
#                 if (i % 4) % 2 == 1:
#                     _list_y[i] = _transform(_list_y[i], 'v')

#         print("len(list_y):", len(list_y))
#         print("len(list_y[0]):", len(list_y[0]))
        
#         y = [torch.cat(_y, dim=0).mean(dim=0, keepdim=True) for _y in list_y]

#         print("y[0].shape:", y[0].shape)

#         if len(y) == 1: y = y[0]

#         return y


#     def forward(self, x):
#         if self.ensemble:
#             print("forward ensemble")
#             out =  self.forward_ensemble(x, forward_function=self.model)
#         else:
#             # print("self.model(x)")
#             out = self.model(x)
#         return out