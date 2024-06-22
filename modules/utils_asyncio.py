import traceback
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Generator, Coroutine

from functools import partial
import asyncio
from asyncio import AbstractEventLoop, Future
import inspect

from modules.logs import import_track, get_logger; import_track(__file__, fp=True); log = get_logger(__name__)  # noqa: E702


def get_coro(obj):
    if inspect.iscoroutine(obj):
        return obj

    elif isinstance(obj, partial) and inspect.iscoroutinefunction(obj.func):
        return obj()

    else:
        raise Exception('obj is not coroutine')


def get_next_generator_result(gen: Generator) -> tuple[Any, bool]:
    """
    Because StopIteration interacts badly with generators and cannot be raised into a Future
    """
    try:
        result = next(gen)
        return result, False
    except StopIteration:
        return None, True



@dataclass(slots=True)
class CoroHandler: # TODO move to au.aio.__init__?
    loop: asyncio.AbstractEventLoop = field(default_factory=asyncio.get_event_loop)
    
    
    ##############
    # Start / stop
    def run_loop_forever(self):
        return self.loop.run_forever()
    
    
    def run(self, coro):
        return self.loop.run_until_complete(coro)
        
    
    def finish_pending(self):
        pending = asyncio.all_tasks(loop=self.loop)
        if pending:
            return asyncio.wait(pending, return_when=asyncio.ALL_COMPLETED)
        return asyncio.sleep(0)
    
    
    async def close_loop(self):
        def __end_background_loop(loop):
            'This might run from another thread'
            tasks = {t for t in asyncio.all_tasks(loop=loop) if not t.done()}
            if not tasks:
                return

            for task in tasks:
                task.cancel()
            loop.stop()

        self.loop.call_soon_threadsafe(__end_background_loop, self.loop)
        await self.loop.shutdown_asyncgens()
        
    
    ############
    # Main tasks
    async def on_error(self, coro, exc, extra_message:str|None=None):
        log.warning(f'Error in coro {coro}')
        traceback_str = ''.join(traceback.format_tb(exc.__traceback__))
        print(traceback_str)
        log.warning('')
        
        log.critical(exc)
        if extra_message:
            log.warning(extra_message)
            
        log.warning('')
        log.warning('')
    
    
    async def _run_wrapped_coro(self, coro, extra_message=None):
        try:
            result = await coro
            return result
        
        except asyncio.CancelledError:
            return None
        
        except Exception as e:
            try:
                await self.on_error(coro, e, extra_message=extra_message)
                
            except asyncio.CancelledError:
                pass
            
            except Exception as e:
                print(traceback.format_tb(e.__traceback__))
                log.critical(f'CoroHandler.on_error failed to run: {e}')
                
            return None
        
        
    def wrap_for_timeout(self, coro, timeout=0, shield=False) -> Coroutine[Any, Any, Any]:
        if timeout > 0:
            if shield:
                coro = asyncio.shield(coro)
            coro = asyncio.wait_for(coro, timeout=timeout)
        return coro
        
    
    async def run_safe(self, coro, extra_message=None, timeout=0, shield=False):
        wrapped = self._run_wrapped_coro(coro, extra_message=extra_message)
        wrapped = self.wrap_for_timeout(wrapped, timeout=timeout, shield=shield)
        return await wrapped
    
    
    def create_task_safe(self, coro, extra_message=None, timeout=0, shield=False):
        wrapped = self.run_safe(coro, extra_message=extra_message, timeout=timeout, shield=shield)
        return asyncio.run_coroutine_threadsafe(wrapped, loop=self.loop)
    
    
    #################
    # Run in executor
    async def run_in_executor(self, func, *args) -> Future:
        """
        Runs a blocking function in a new thread so it can be awaited.
        """
        # TODO is error handling needed?
        return await self.loop.run_in_executor(None, func, *args) 


    async def generate_in_executor(self, func, *args):
        """
        Converts a blocking generator to an async one
        """
        # init the generator
        gen = await self.loop.run_in_executor(None, func, *args)

        while True:
            result, is_done = await self.loop.run_in_executor(None, get_next_generator_result, gen)
            if is_done:
                break

            yield result



async def generate_in_executor(partial: partial, *args, loop: AbstractEventLoop|None = None) -> AsyncGenerator[Any, Any]:
    """
    Converts a blocking generator to an async one
    """
    loop = loop or asyncio.get_running_loop()
    gen = await loop.run_in_executor(None, partial, *args)

    while True:
        result, is_done = await loop.run_in_executor(None, get_next_generator_result, gen)
        if is_done:
            break

        yield result


async def run_in_executor(partial: partial, *args, loop: AbstractEventLoop|None = None) -> Future:
    """
    Runs a blocking function in a new thread so it can be awaited.
    """
    loop = loop or asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial, *args)
