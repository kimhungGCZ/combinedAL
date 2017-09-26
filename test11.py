import trollius
from trollius import From

@trollius.coroutine
def greet_every_two_seconds():
    while True:
        print('Hello World')
        yield From(trollius.sleep(2))

loop = trollius.get_event_loop()
loop.run_until_complete(greet_every_two_seconds())