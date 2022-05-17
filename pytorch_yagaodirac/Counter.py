import math

class Counter:
    def __init__(self, start_from = 5, every = None):
        self.next = start_from
        if None == every:
            every = 1
            pass
        assert every>=1
        self.every = every
        pass
    def get(self, current):
        if current>= self.next:
            self.next = current
            self.next += self.every
            return True
            pass
        return False
        pass
    pass




class Counter_log:
    def __init__(self, max_step_length = 1000, *, min_step_length = 1, min_step_times = 10):
        assert max_step_length>min_step_length
        self.max_step_length = max_step_length
        self.min_step_length = min_step_length
        self.min_step_times = min_step_times
        self.next = min_step_length
        pass
    def get(self, current):
        if current >= self.next:
            if current<self.min_step_length * self.min_step_times:
                #Still in the dence part.
                for _ in range(self.min_step_times):
                    self.next += self.min_step_length
                    if self.next >= current:
                        break
                        pass
                    pass
                pass#if
            else:
                self.next = self._calc_next(current)
                pass#else
            return True
            pass
        return False
        pass
    def _calc_next(self, x):
        unit = 10**math.floor(math.log10(x))
        if unit> self.max_step_length:
            unit = 10**math.floor(math.log10(self.max_step_length))
            pass
        first_digit = x//unit
        first_digit = first_digit+1
        result = first_digit * unit
        return result
        pass
    pass


if 0:
    c = Counter_log(max_step_length=100, min_step_length=1)#or 3)
    result = c.get(1)
    result = c.get(1)
    result = c.get(2)
    result = c.get(5)
    result = c.get(10)
    result = c.get(10)
    result = c.get(11)
    result = c.get(19)
    result = c.get(20)
    result = c.get(1000)
    result = c.get(1100)
    result = c.get(1100)
    result = c.get(1200)
    result = c.get(1200)
    jdfskl = 345798
    pass







