import colorama
from colorama import Fore, Back, Style
import sys
import logging
import os
log = logging.getLogger('bot')
colorama.init()

COLORS = {
    'WARNING': f'{Fore.LIGHTRED_EX} WARN [{{name}}]: ',
    'INFO': f'{Fore.GREEN} INFO [{{name}}]: {Fore.LIGHTGREEN_EX}',
    'DEBUG': f'{Fore.LIGHTCYAN_EX}DEBUG [{{name}}]: ',
    'CRITICAL': f'{Back.RED}{Fore.WHITE} CRIT [{{name}}]: ',
    'ERROR': f'{Fore.RED}ERROR [{{name}}]: ',
    'TEST': f'{Fore.YELLOW} TEST [{{name}}]: {Fore.LIGHTYELLOW_EX}'
}

TEST_LEVEL = 51

logging.addLevelName(TEST_LEVEL, "TEST")

class ColoredFormatter(logging.Formatter):
    def __init__(self, msg, use_color = True, **kw):
        logging.Formatter.__init__(self, msg, **kw)
        self.use_color = use_color

    def format(self, record):
        levelname = record.levelname
        name = record.name
        if self.use_color and levelname in COLORS:
            record.levelname = COLORS[levelname].format(levelname=levelname, name=name)

        f = logging.Formatter.format(self, record)
        record.levelname = levelname
        return f

def test(message, *args, **kws):
    log._log(TEST_LEVEL, message, args, **kws)

def get_logger(name):
    return log.getChild(name)

log.test = test # type: ignore
log_level = logging.INFO

log_formatter = ColoredFormatter(f'{Fore.BLACK}{Back.WHITE}%(asctime)s.%(msecs)03d{Back.BLACK} {Back.LIGHTBLACK_EX}#%(lineno)-5d{Style.RESET_ALL} {Fore.LIGHTWHITE_EX}%(levelname)s%(message)s{Style.RESET_ALL}', datefmt='%H:%M:%S')
log.setLevel(log_level)


def add_file_handler(level=logging.DEBUG, fp="latest.log", **kw):
    log_formatter_file = ColoredFormatter('[%(asctime)-15s] %(levelname)-8s #%(lineno)-5d (%(name)s) %(module)s.%(funcName)s -> %(message)s', use_color=False)
    file_handler = logging.FileHandler(fp, encoding='utf-8', **kw)
    file_handler.setFormatter(log_formatter_file)
    file_handler.setLevel(level)
    log.addHandler(file_handler)
    return file_handler, log_formatter_file

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(log_level)
console_handler.setFormatter(log_formatter)
log.addHandler(console_handler)


logging.getLogger('asyncio').setLevel(logging.WARNING)
log_file_handler, log_file_formatter = add_file_handler(fp='discord.log', mode='w')


################
# import tracker
root_path_ = os.path.dirname(os.path.abspath(__file__)).rsplit(os.sep,2)[0].lower()
_import_log = get_logger(__name__)
def import_track(string, fp=False):
    if fp:# and root_path_ in string:
        string = string.lower().split(root_path_,1)[1]
        string = string.lstrip(os.sep)
        string = string.rsplit('.',1)[0]
        string = string.replace(os.sep,'.')

    _import_log.debug(f'IMPORT {string}')

