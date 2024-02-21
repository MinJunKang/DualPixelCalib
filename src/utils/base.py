
import os
import sys
import shutil


def create_dir(path, exist_ok=True):
    if not exist_ok and path.exists():
        shutil.rmtree(str(path))
    os.makedirs(str(path), exist_ok=True)
    return path


# condition checker
def check_condition(condition, warner):
    if not condition:
        raise NotImplementedError(warner)
    else:
        return True


def error_handler(condition, expression, name, stop=False):
    
    '''
    :param condition: condition to check
    :param expression: error message
    :param name: location of error, use __name__ in this place
    :param stop: if condition is wrong, stop the process
    :return:
    '''

    try:
        assert condition
    except:
        if stop:
            raise NotImplementedError('%s : %s\n' % (name, expression))
        else:
            print('%s : %s\n' % (name, expression))
    
    
def option_check(value, options=None):
    error_handler(value in options, "option_check failed : %s" % value, __name__, True)