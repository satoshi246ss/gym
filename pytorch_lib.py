import inspect
import pickle
from copy import deepcopy
from importlib import machinery
import torch.nn.functional as F
from torch import nn

def load_model(filename):
    ''' モデル・重みの読み込み '''
    with open(filename, 'rb') as f:
        state = pickle.load(f)
    module = machinery.SourceFileLoader(
        state['module_path'], state['module_path']).load_module()
    args, kwargs = state['args'], state['kwargs']
    model = getattr(module, state['class_name'])(*args, **kwargs)
    model.load_state_dict(state['state_dict'])
    return model

def save_model(model, filename, args=[], kwargs={}):
    ''' モデル・重みの保存 '''
    model_cpu = deepcopy(model).cpu()
    state = {'module_path': inspect.getmodule(model, _filename=True).__file__,
             'class_name': model.__class__.__name__,
             'state_dict': model_cpu.state_dict(),
             'args': args,
             'kwargs': kwargs}
    with open(filename, 'wb') as f:
        pickle.dump(state, f)

if __name__ == '__main__':
    # モデル定義
    model = nn(128, 2)
    print(model)
    # 保存
    save_model(model, 'net.pkl', args=[128, 2])
    # 読み込み
    model = load_model('net.pkl')
    print(model)