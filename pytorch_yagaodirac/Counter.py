class Counter:
    def __init__(self, start_from = 5, every = -1):
        self.next = start_from
        self.every = every
        if every<=0:
            self.every = start_from
            pass
        pass
    def get(self, current):
        if current>= self.next:
            self.next += self.every
            return True
            pass
        return False
        pass
    pass
