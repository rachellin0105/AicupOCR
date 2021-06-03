from DB.concern.config import Configurable, State
from DB.concern.log import Logger
from DB.structure.builder import Builder
from DB.structure.representers import *
from DB.structure.measurers import *
from DB.structure.visualizers import *
from DB.data.data_loader import *
from DB.data import *
from DB.training.model_saver import ModelSaver
from DB.training.checkpoint import Checkpoint
from DB.training.optimizer_scheduler import OptimizerScheduler


class Structure(Configurable):
    builder = State()
    representer = State()
    measurer = State()
    visualizer = State()

    def __init__(self, **kwargs):
        self.load_all(**kwargs)

    @property
    def model_name(self):
        return self.builder.model_name


class TrainSettings(Configurable):
    data_loader = State()
    model_saver = State()
    checkpoint = State()
    scheduler = State()
    epochs = State(default=10)

    def __init__(self, **kwargs):
        kwargs['cmd'].update(is_train=True)
        self.load_all(**kwargs)
        if 'epochs' in kwargs['cmd']:
            self.epochs = kwargs['cmd']['epochs']


class ValidationSettings(Configurable):
    data_loaders = State()
    visualize = State()
    interval = State(default=100)
    exempt = State(default=-1)

    def __init__(self, **kwargs):
        kwargs['cmd'].update(is_train=False)
        self.load_all(**kwargs)

        cmd = kwargs['cmd']
        self.visualize = cmd['visualize']


class EvaluationSettings(Configurable):
    data_loaders = State()
    visualize = State(default=True)
    resume = State()

    def __init__(self, **kwargs):
        self.load_all(**kwargs)


class EvaluationSettings2(Configurable):
    structure = State()
    data_loaders = State()

    def __init__(self, **kwargs):
        self.load_all(**kwargs)


class ShowSettings(Configurable):
    data_loader = State()
    representer = State()
    visualizer = State()

    def __init__(self, **kwargs):
        self.load_all(**kwargs)


class Experiment(Configurable):
    structure = State(autoload=False)
    train = State()
    validation = State(autoload=False)
    evaluation = State(autoload=False)
    logger = State(autoload=True)

    def __init__(self, **kwargs):
        self.load('structure', **kwargs)

        cmd = kwargs.get('cmd', {})
        if 'name' not in cmd:
            cmd['name'] = self.structure.model_name

        self.load_all(**kwargs)
        self.distributed = cmd.get('distributed', False)
        self.local_rank = cmd.get('local_rank', 0)

        if cmd.get('validate', False):
            self.load('validation', **kwargs)
        else:
            self.validation = None
